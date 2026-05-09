import sys
import os
from typing import Callable, Any
import math
from collections import deque
from flight_stack_msgs.srv import SaveFrame, CalculateDistance
from std_srvs.srv import SetBool
from std_msgs.msg import Float32
from px4_msgs.msg import ManualControlSetpoint
from ultralytics import YOLO

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from pyorbbecsdk import (  # pylint: disable=no-name-in-module
    Config,
    Pipeline,
    FrameSet,
    OBSensorType,
    OBFormat,
    OBError,
    OBPropertyID,
    VideoStreamProfile,
    transformation2dto3d,
    OBPoint2f,
)
from computer_vision.utils import frame_to_bgr_image


class Stream(Node):
    def __init__(self):
        super().__init__("Stream")

        self.declare_parameter("cam_pos_n", 0.26)
        self.declare_parameter("cam_pos_e", 0.0)
        self.declare_parameter("cam_pos_d", 0.055)
        self.declare_parameter("cam_pitch", 0.0)
        
        self.declare_parameter("pump_pos_n", 0.22)
        self.declare_parameter("pump_pos_e", 0.0)
        self.declare_parameter("pump_pos_d", 0.15)
        
        self.declare_parameter("nozzle_length", 0.05)
        self.declare_parameter("v_0", 10.0)
        self.declare_parameter("dist_threshold", 150.0)
        
        self.frame = 0

        self._video = None
        self._video_read: Callable[[], tuple[bool, Any]] | None = None
        self._frame_size: tuple[int, int] | None = None
        self._frame_rate: int | None = None
        self._frame_buffer = {}  # dictionary keyed by frame_id
        self._saved_frames = {}  # explicitly saved frames for distance calculation
        self._depth_min_history = deque(maxlen=30)
        self._depth_max_history = deque(maxlen=30)
        self._last_depth_data = None
        self._last_depth_frame_info = None
        self._overlay_depth = "--overlay-depth" in sys.argv
        self._flip = "--flip" in sys.argv

        self._pump_angle = 0

        # Load YOLO model
        model_path = ""
        if torch.cuda.is_available():
            print("CUDA is available. Attempting to load TensorRT engine.")
            model_path = os.path.join(os.path.dirname(__file__), "./best.engine")
        else:
            print("CUDA not available. Loading standard PyTorch model.")
            model_path = os.path.join(os.path.dirname(__file__), "./best.pt")
        try:
            # TensorRT models (.engine) natively run on the GPU they were compiled for.
            self._model = YOLO(model_path, task='detect')
        except Exception as e:
            print(f"Failed to load YOLO engine: {e}")
            self._model = None

        self.create_subscription(ManualControlSetpoint, '/fmu/out/manual_control_setpoint', self._servo_callback, qos_profile_sensor_data)
        self.create_subscription(ManualControlSetpoint, '/fmu/in/manual_control_input', self._servo_callback, qos_profile_sensor_data)

        self._x_error_pub = self.create_publisher(Float32, '/uas/cv/x_error', 10)

        self.create_service(
            CalculateDistance, "/uas/cv/calculate_distance", self._distance_callback
        )
        self.create_service(SaveFrame, "/uas/cv/save_frame", self._save_frame_callback)
        self.create_service(SetBool, "/uas/cv/toggle_overlay_depth", self._toggle_overlay_callback)

        config = Config()
        pipeline = Pipeline()
        try:
            device = pipeline.get_device()
            if self._flip:
                try:
                    device.set_int_property(OBPropertyID.OB_PROP_COLOR_ROTATE_INT, 180)
                    device.set_int_property(OBPropertyID.OB_PROP_DEPTH_ROTATE_INT, 180)
                except OBError as e:
                    print("Failed to set rotation properties:", e)

            profile_list = pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            try:
                color_profile: VideoStreamProfile = (
                    profile_list.get_video_stream_profile(
                        1280, 800, OBFormat.RGB, 30
                    )
                )
            except OBError as e:
                print(e)
                color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
            config.enable_stream(color_profile)

            # Enable depth sensor
            depth_profile_list = pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            try:
                depth_profile: VideoStreamProfile = (
                    depth_profile_list.get_video_stream_profile(
                        1280, 800, OBFormat.Y16, 30
                    )
                )
            except OBError as e:
                print(e)
                depth_profile = (
                    depth_profile_list.get_default_video_stream_profile()
                )
            print("depth profile: ", depth_profile)
            config.enable_stream(depth_profile)

        except Exception as e:
            print(e)
            return
        pipeline.start(config)

        def read_orbbec_frame():
            frames: FrameSet = pipeline.wait_for_frames(1000)
            if frames is None:
                return False, None
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                return False, None

            # Get central 3D coordinate
            depth_width = depth_frame.get_width()
            depth_height = depth_frame.get_height()

            if depth_frame.get_data_size() == depth_width * depth_height * 2:
                color_stream_profile = color_frame.get_stream_profile()
                depth_stream_profile = depth_frame.get_stream_profile()
                depth_intrinsics = (
                    depth_stream_profile.as_video_stream_profile().get_intrinsic()
                )
                extrinsic = depth_stream_profile.get_extrinsic_to(
                    color_stream_profile
                )

                depth_data = np.frombuffer(
                    depth_frame.get_data(), dtype=np.uint16
                ).reshape(depth_height, depth_width)

                # store latest depth info for save_frame service
                self._last_depth_data = depth_data.copy()
                self._last_depth_frame_info = {
                    "depth_intrinsics": depth_intrinsics,
                    "extrinsic": extrinsic,
                    "depth_width": depth_width,
                    "depth_height": depth_height,
                }

            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                return False, None

            return True, color_image

        self._video_read = read_orbbec_frame
        self._frame_size = (1280, 800)
        self._frame_rate = 30

        self._stream = cv2.VideoWriter(
            'appsrc ! timecodestamper ! webrtcsink forward-metas="timecode" name=ws enable-control-data-channel=true',
            cv2.CAP_GSTREAMER,
            0,
            self._frame_rate,
            self._frame_size,
            True,
        )
        # if not self._video.isOpened():
        #     print("Failed to open video")
        #     rclpy.shutdown()

        self.create_timer(0.001, self._publish)

    def _publish(self):
        ret, frame = self._video_read()

        if not ret:
            print("Failed to grab frame.")
            return

        # Draw crosshair at the center
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        if self._model is not None:
            results = list(self._model(frame, conf=0.4, stream=True, verbose=False))
            frame = results[0].plot()
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                obj_center_x = (x1 + x2) / 2.0
                x_error = float(obj_center_x - center_x)
                
                msg = Float32()
                msg.data = x_error
                self._x_error_pub.publish(msg)

        if self._overlay_depth and self._last_depth_data is not None:
            depth_data = self._last_depth_data
            depth_width = self._last_depth_frame_info["depth_width"]
            depth_height = self._last_depth_frame_info["depth_height"]

            # Use OpenCV's built-in mask and minMaxLoc which are C++ optimized
            valid_mask = cv2.compare(depth_data, 0, cv2.CMP_GT)
            min_val, max_val, _, _ = cv2.minMaxLoc(depth_data, mask=valid_mask)

            if max_val > 0:
                self._depth_min_history.append(min_val)
                self._depth_max_history.append(max_val)

            if len(self._depth_min_history) > 0:
                avg_min = np.mean(self._depth_min_history)
                avg_max = np.mean(self._depth_max_history)
                if avg_max <= avg_min:
                    avg_max = avg_min + 1

                alpha_scale = 255.0 / (avg_max - avg_min)
                beta_scale = -avg_min * alpha_scale
                depth_normalized = cv2.convertScaleAbs(depth_data, alpha=alpha_scale, beta=beta_scale)
            else:
                depth_normalized = np.zeros((depth_height, depth_width), dtype=np.uint8)

            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_colormap = cv2.bitwise_and(depth_colormap, depth_colormap, mask=valid_mask)

            if frame.shape[:2] != depth_colormap.shape[:2]:
                depth_colormap = cv2.resize(
                    depth_colormap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
                )

            alpha = 0.5
            frame = cv2.addWeighted(frame, 1 - alpha, depth_colormap, alpha, 0)


        # Draw actual hit projection
        if self._last_depth_data is not None and self._last_depth_frame_info is not None:
            depth_data = self._last_depth_data
            depth_w = self._last_depth_frame_info["depth_width"]
            depth_h = self._last_depth_frame_info["depth_height"]
            intrinsics = self._last_depth_frame_info["depth_intrinsics"]
            extrinsic = self._last_depth_frame_info["extrinsic"]
            
            cam_n = self.get_parameter("cam_pos_n").value
            cam_e = self.get_parameter("cam_pos_e").value
            cam_d = self.get_parameter("cam_pos_d").value
            cam_pitch = self.get_parameter("cam_pitch").value
            
            pump_n = self.get_parameter("pump_pos_n").value
            pump_e = self.get_parameter("pump_pos_e").value
            pump_d = self.get_parameter("pump_pos_d").value
            nozzle_length = self.get_parameter("nozzle_length").value
            
            v_0 = self.get_parameter("v_0").value
            dist_threshold = self.get_parameter("dist_threshold").value

            # Precompute trigonometric constants
            cos_cam_pitch = math.cos(cam_pitch)
            sin_cam_pitch = math.sin(cam_pitch)
            cos_pump = math.cos(self._pump_angle)
            sin_pump = math.sin(self._pump_angle)
            tan_pump = math.tan(self._pump_angle)

            # Target pixels: 4 pixels wide slice at center
            slice_w = 4
            x_start = max(0, (depth_w // 2) - (slice_w // 2))
            x_end = min(depth_w, x_start + slice_w)

            # Collect depth pixels and extract 3D coords
            pts_3d_raw = []
            pixel_coords = []
            for y in range(depth_h):
                for x in range(x_start, x_end):
                    d_val = depth_data[y, x]
                    if d_val > 0:
                        pt_3d = transformation2dto3d(OBPoint2f(float(x), float(y)), float(d_val), intrinsics, extrinsic)
                        pts_3d_raw.append([pt_3d.x / 1000.0, pt_3d.y / 1000.0, pt_3d.z / 1000.0])
                        pixel_coords.append((x, y))
            
            matched_pixels = []
            if len(pts_3d_raw) > 0:
                pts_3d_arr = np.array(pts_3d_raw)
                
                # Vectorize NED transformation
                opt_x = pts_3d_arr[:, 0]
                opt_y = pts_3d_arr[:, 1]
                opt_z = pts_3d_arr[:, 2]
                
                n_pt = opt_z * cos_cam_pitch + opt_y * sin_cam_pitch + cam_n
                e_pt = opt_x + cam_e
                d_pt = opt_y * cos_cam_pitch - opt_z * sin_cam_pitch + cam_d
                
                pts_ned = np.column_stack((n_pt, e_pt, d_pt))

                # Gradient descent approximation for closest point on water path
                t_m = 5.0  # Start ~5m out
                step = 3.0 # Initial search step size in meters

                pump_offset_n = pump_n + nozzle_length * cos_pump
                pump_offset_d = pump_d - nozzle_length * sin_pump

                for _ in range(10):
                    if step < 0.05:  # Early termination
                        break
                        
                    t_cands = [max(0.0, t_m - step), t_m, t_m + step]
                    min_dists = []
                    
                    # Evaluate trajectory points
                    for t in t_cands:
                        dx = v_0 * t * math.cos(self._pump_angle)
                        dz = -v_0 * t * math.sin(self._pump_angle) + 0.5 * 9.8 * (t**2)
                        pn = pump_n + nozzle_length * math.cos(self._pump_angle) + dx
                        pe = pump_e
                        pd = pump_offset_d + dz
                        
                        pt_traj = np.array([pn, pe, pd])
                        # Distance to all points
                        dist = np.min(np.linalg.norm(pts_ned - pt_traj, axis=1))
                        min_dists.append(dist)
                    
                    # Update t_m to the best candidate
                    best_idx = np.argmin(min_dists)
                    t_m = t_cands[best_idx]
                    
                    # Reduce step size for next iteration
                    step *= 0.5
                
                # Find all points within threshold using the final best t_m
                dx = v_0 * t_m * math.cos(self._pump_angle)
                dz = -v_0 * t_m * math.sin(self._pump_angle) + 0.5 * 9.8 * (t_m**2)
                pn = pump_n + nozzle_length * math.cos(self._pump_angle) + dx
                pe = pump_e
                pd = pump_d - nozzle_length * math.sin(self._pump_angle) + dz
                best_traj_pt = np.array([pn, pe, pd])
                
                dists = np.linalg.norm(pts_ned - best_traj_pt, axis=1)
                num_points = min(10, len(dists))
                valid_indices = np.argsort(dists)[:num_points]
                
                scale_x = width / depth_w
                scale_y = height / depth_h
                for idx in valid_indices:
                    x, y = pixel_coords[idx]
                    c_x = int(x * scale_x)
                    c_y = int(y * scale_y)
                    matched_pixels.append((c_x, c_y))
            
            for (cx, cy) in matched_pixels:
                cv2.drawMarker(frame, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)

        # Crosshair parameters
        crosshair_length = 20
        crosshair_color = (0, 255, 0)  # Green
        thickness = 2

        # Horizontal line
        cv2.line(
            frame,
            (center_x - crosshair_length, center_y),
            (center_x + crosshair_length, center_y),
            crosshair_color,
            thickness,
        )
        # Vertical line
        cv2.line(
            frame,
            (center_x, center_y - crosshair_length),
            (center_x, center_y + crosshair_length),
            crosshair_color,
            thickness,
        )

        self._stream.write(frame)
        self.frame += 1

        if (
            self._last_depth_data is not None
            and self._last_depth_frame_info is not None
        ):
            self._frame_buffer[self.frame] = {
                "depth_data": self._last_depth_data,
                "depth_intrinsics": self._last_depth_frame_info["depth_intrinsics"],
                "extrinsic": self._last_depth_frame_info["extrinsic"],
                "depth_width": self._last_depth_frame_info["depth_width"],
                "depth_height": self._last_depth_frame_info["depth_height"],
            }
            if len(self._frame_buffer) > 1000:
                oldest_key = min(self._frame_buffer.keys())
                del self._frame_buffer[oldest_key]

    def _get_3d_point(self, x, y, frame_data):
        depth_data = frame_data["depth_data"]
        depth_width = frame_data["depth_width"]
        depth_height = frame_data["depth_height"]

        # Ensure coordinates are within bounds
        if x < 0 or x >= depth_width or y < 0 or y >= depth_height:
            return None

        depth_value = depth_data[y, x]

        if depth_value == 0:
            angles = [i * math.pi / 4 for i in range(8)]
            non_zero_depths = []
            max_radius = 50  # Prevent searching too far

            for angle in angles:
                dx = math.cos(angle)
                dy = math.sin(angle)

                for r in range(1, max_radius):
                    nx = int(round(x + r * dx))
                    ny = int(round(y + r * dy))

                    if nx < 0 or nx >= depth_width or ny < 0 or ny >= depth_height:
                        break

                    val = depth_data[ny, nx]
                    if val > 0:
                        non_zero_depths.append(val)
                        break

            if len(non_zero_depths) > 0:
                depth_value = np.mean(non_zero_depths)

        if depth_value > 0:
            return transformation2dto3d(
                OBPoint2f(float(x), float(y)),
                float(depth_value),
                frame_data["depth_intrinsics"],
                frame_data["extrinsic"],
            )
        return None

    def _save_frame_callback(self, request, response):
        target_frame = request.frame_id

        if target_frame in self._frame_buffer:
            self._saved_frames[target_frame] = self._frame_buffer[target_frame]
            response.success = True
            response.message = f"Frame {target_frame} saved."
            response.frame_id = target_frame
        else:
            response.success = False
            response.message = f"Frame {target_frame} not found in buffer."
            response.frame_id = target_frame

        return response

    def _distance_callback(self, request, response):
        target_frame = request.frame_id
        x1, y1 = request.x1, request.y1
        x2, y2 = request.x2, request.y2

        frame_data = self._saved_frames.get(target_frame)

        if frame_data is None:
            response.success = False
            response.message = f"Frame {target_frame} not found in saved frames."
            response.distance = -1.0
            return response

        p1 = self._get_3d_point(x1, y1, frame_data)
        p2 = self._get_3d_point(x2, y2, frame_data)

        if p1 and p2:
            distance = math.sqrt(
                (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
            )
            response.success = True
            response.message = f"Calculated distance: {distance:.2f} mm"
            response.distance = distance
            print(
                f"Distance between ({x1}, {y1}) and ({x2}, {y2}) at frame {target_frame}: {distance:.2f} mm"
            )
        else:
            response.success = False
            response.message = f"Could not calculate distance (invalid depth for one or both points at frame {target_frame})"
            response.distance = -1.0
            print(response.message)

        return response

    def _toggle_overlay_callback(self, request, response):
        self._overlay_depth = request.data
        response.success = True
        response.message = f"Depth overlay set to {self._overlay_depth}"
        return response

    def _servo_callback(self, msg: ManualControlSetpoint):
        if not math.isnan(msg.aux3):
            self._pump_angle = msg.aux3 * math.radians(32.0)

    def shut_down_cv(self):
        if self._stream is not None:
            self._stream.release()


def main(args=None):
    rclpy.init(args=args)
    stream = Stream()
    rclpy.spin(stream)

    stream.shut_down_cv()
    cv2.destroyAllWindows()
    stream.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
