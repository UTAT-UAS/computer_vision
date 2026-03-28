import os
import math

from scipy.spatial.transform import Rotation
import cv2
# from aiortc import VideoStreamTrack

from ultralytics import YOLO
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from geometry_msgs.msg import Point


OBJECT_SIZE = 0.8128


class CircularBuffer:
    def __init__(self, size: int = 10) -> None:
        self.l = [None for i in range(size)]
        self._size = size
        self._i = 0

    @property
    def size(self) -> int:
        return self._size

    def push(self, obj) -> None:
        self.l[self._i] = obj
        self._i += 1
        self._i = self._i % self._size


class Detector(Node):
    def __init__(self):
        super().__init__("Detector")

        self.frame = 0

        self._position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._position_cb,
            QoSPresetProfiles.SENSOR_DATA.value,
        )
        self._position = VehicleLocalPosition()
        self._position.x = 0.0
        self._position.y = 0.0
        self._position.z = 0.0
        self._attitude_subscriber = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self._attitude_cb,
            QoSPresetProfiles.SENSOR_DATA.value,
        )
        self._attitude = VehicleAttitude()
        self._attitude.q = [-1.0, 0.0, 0.0, 0.0]

        self._position_pub = self.create_publisher(
            Point, "/uas/cv/position", QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

        torch.cuda.set_device(0)
        model_path = os.path.join(os.path.dirname(__file__), "./landing_pad.pt")

        if torch.cuda.is_available():
            print("CUDA ON")
            torch.cuda.set_device(0)
            self._model = YOLO(model_path).to("cuda")
        else:
            print("CUDA OFF")
            self._model = YOLO(model_path)

        self._video = cv2.VideoCapture(
            "udpsrc port=5600 ! application/x-rtp,payload=96,encoding-name=H264 ! rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink drop=true sync=false",
            cv2.CAP_GSTREAMER,
        )
        # self._video = cv2.VideoCapture(0)
        # self._video.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self._stream = cv2.VideoWriter(
            # "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000",
            'appsrc ! timecodestamper ! webrtcsink forward-metas="timecode" name=ws enable-control-data-channel=true',
            cv2.CAP_GSTREAMER,
            0,
            20,
            (1280, 960),
            # (640, 480),
            True,
        )
        if not self._video.isOpened():
            print("Failed to open video")
            rclpy.shutdown()

        self._bufferx = CircularBuffer(10)
        self._buffery = CircularBuffer(10)
        self.create_timer(0.05, self._detect)

    def _position_cb(self, msg: VehicleLocalPosition) -> None:
        self._position = msg

    def _attitude_cb(self, msg: VehicleAttitude) -> None:
        self._attitude = msg

    def _detect(self):
        ret, frame = self._video.read()

        if not ret:
            print("Failed to grab frame.")
            return

        results = self._model.predict(frame, stream=True, verbose=False)
        for r in results:
            frame = r.plot()
            original = r.orig_shape
            if len(r.boxes) > 0:
                box = list(r.boxes[0].xyxy[0].to("cpu"))
                _, _, width, height = list(r.boxes[0].xywh[0].to("cpu"))
                center = original[1] / 2, original[0] / 2
                centroid = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                x = center[1] - centroid[1]  # NED body frame
                y = centroid[0] - center[0]  # NED body frame
                pixels_per_meter = width / OBJECT_SIZE
                self._bufferx.push(x / pixels_per_meter)
                self._buffery.push(y / pixels_per_meter)

                rot = Rotation.from_quat(  # PX4 is (w, x, y, z)
                    [
                        self._attitude.q[1],
                        self._attitude.q[2],
                        self._attitude.q[3],
                        self._attitude.q[0],
                    ]
                )
                rot_euler = rot.as_euler("zyx")
                angle = rot_euler[0]
                angle = angle + 2 * math.pi if angle < 0 else angle
                x = self._avg_buffer(self._bufferx)
                y = self._avg_buffer(self._buffery)

                # print(math.degrees(angle))
                rx = x * math.cos(angle) - y * math.sin(angle)
                ry = y * math.cos(angle) + x * math.sin(angle)
                point = Point(x=self._position.x + rx, y=self._position.y + ry)
                self._position_pub.publish(point)
            break

        fh, fw = frame.shape[:2]  # Slicing [:2] gets only height and width

        cv2.line(
            frame,
            (fw // 2, 0),
            (fw // 2, fh),
            (0, 0, 255),
            3,
        )
        cv2.line(
            frame,
            (0, fh // 2),
            (fw, fh // 2),
            (0, 255, 0),
            3,
        )

        print(f"Frame {self.frame}")
        self.frame += 1
        self._stream.write(frame)

    def _avg_buffer(self, buffer: CircularBuffer) -> float:
        total = 0
        for b in buffer.l:
            if b is not None:
                total += b
        return float(total / buffer.size)

    def shut_down_cv(self):
        self._video.release()
        self._stream.release()


def main(args=None):
    rclpy.init(args=args)
    detector = Detector()
    rclpy.spin(detector)

    detector.shut_down_cv()
    cv2.destroyAllWindows()
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
