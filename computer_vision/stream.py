import sys
from typing import Callable, Any
import math
from collections import deque
from std_msgs.msg import Int32MultiArray

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from pyorbbecsdk import (  # pylint: disable=no-name-in-module
    Config,
    Pipeline,
    FrameSet,
    OBSensorType,
    OBFormat,
    OBError,
    VideoStreamProfile,
    transformation2dto3d,
    OBPoint2f,
)
from computer_vision.utils import frame_to_bgr_image


class Stream(Node):
    def __init__(self):
        super().__init__("Stream")

        self.frame = 0

        self._video = None
        self._video_read: Callable[[], tuple[bool, Any]] | None = None
        self._frame_size: tuple[int, int] | None = None
        self._frame_rate: int | None = None
        self._frame_buffer = deque(maxlen=1000)
        self._depth_min_history = deque(maxlen=30)
        self._depth_max_history = deque(maxlen=30)

        self.create_subscription(
            Int32MultiArray, "calculate_distance", self._distance_callback, 10
        )

        if "--simulation" in sys.argv:
            self._video = cv2.VideoCapture(
                "udpsrc port=5600 ! application/x-rtp,payload=96,encoding-name=H264 ! rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=true sync=false",
                cv2.CAP_GSTREAMER,
            )
            self._frame_size = (1280, 960)
            self._frame_rate = 20
        elif "--webcam" in sys.argv:
            self._video = cv2.VideoCapture(0)
            self._video.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            self._frame_size = (640, 480)
            self._frame_rate = 20
        elif "--depth" in sys.argv:
            config = Config()
            pipeline = Pipeline()
            try:
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

                    self._frame_buffer.append(
                        {
                            "frame_num": self.frame,
                            "depth_data": depth_data.copy(),
                            "depth_intrinsics": depth_intrinsics,
                            "extrinsic": extrinsic,
                            "depth_width": depth_width,
                            "depth_height": depth_height,
                        }
                    )

                # covert to RGB format
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    print("failed to convert frame to image")
                    return False, None

                if (
                    "--overlay-depth" in sys.argv
                    and depth_frame.get_data_size() == depth_width * depth_height * 2
                ):
                    valid_depths = depth_data[depth_data > 0]
                    if len(valid_depths) > 0:
                        self._depth_min_history.append(np.min(valid_depths))
                        self._depth_max_history.append(np.max(valid_depths))

                    if len(self._depth_min_history) > 0:
                        avg_min = np.mean(self._depth_min_history)
                        avg_max = np.mean(self._depth_max_history)
                        if avg_max <= avg_min:
                            avg_max = avg_min + 1

                        clipped_depth = np.clip(depth_data, avg_min, avg_max)
                        depth_normalized = (
                            (clipped_depth - avg_min) / (avg_max - avg_min) * 255.0
                        ).astype(np.uint8)
                    else:
                        depth_normalized = np.zeros(
                            (depth_height, depth_width), dtype=np.uint8
                        )

                    # Apply false color mapping to the normalized depth
                    depth_colormap = cv2.applyColorMap(
                        depth_normalized, cv2.COLORMAP_JET
                    )

                    # Resize the depth colormap if its dimensions differ from color_image
                    if color_image.shape[:2] != depth_colormap.shape[:2]:
                        depth_colormap = cv2.resize(
                            depth_colormap, (color_image.shape[1], color_image.shape[0])
                        )

                    # Overlay the depth map on top of the color image using simple alpha blending
                    alpha = 0.5
                    color_image = cv2.addWeighted(
                        color_image, 1 - alpha, depth_colormap, alpha, 0
                    )

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
        # ret, frame = self._video.read()
        ret, frame = self._video_read()

        if not ret:
            print("Failed to grab frame.")
            return

        # Draw crosshair at the center
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

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

    def _get_3d_point(self, x, y, frame_data):
        depth_data = frame_data["depth_data"]
        depth_width = frame_data["depth_width"]
        depth_height = frame_data["depth_height"]

        # Ensure coordinates are within bounds
        if x < 0 or x >= depth_width or y < 0 or y >= depth_height:
            return None

        depth_value = depth_data[y, x]

        if depth_value == 0:
            window_size = 30
            half_window = window_size // 2
            y_start = max(0, y - half_window)
            y_end = min(depth_height, y + half_window + 1)
            x_start = max(0, x - half_window)
            x_end = min(depth_width, x + half_window + 1)

            surrounding = depth_data[y_start:y_end, x_start:x_end]
            non_zero_depths = surrounding[surrounding > 0]
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

    def _distance_callback(self, msg):
        if len(msg.data) != 5:
            print("Expected 5 values in Int32MultiArray: [frame_num, x1, y1, x2, y2]")
            return

        target_frame, x1, y1, x2, y2 = msg.data

        frame_data = None
        for f in self._frame_buffer:
            if f["frame_num"] == target_frame:
                frame_data = f
                break

        if frame_data is None:
            print(
                f"Frame {target_frame} not found in buffer (max {self._frame_buffer.maxlen} frames)."
            )
            return

        p1 = self._get_3d_point(x1, y1, frame_data)
        p2 = self._get_3d_point(x2, y2, frame_data)

        if p1 and p2:
            distance = math.sqrt(
                (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
            )
            print(
                f"Distance between ({x1}, {y1}) and ({x2}, {y2}) at frame {target_frame}: {distance:.2f} mm"
            )
        else:
            print(
                f"Could not calculate distance (invalid depth for one or both points at frame {target_frame})"
            )

    def shut_down_cv(self):
        self._video.release()
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
