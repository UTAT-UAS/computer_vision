import sys
from typing import Callable, Any

import cv2

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
                        profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
                    )
                except OBError as e:
                    print(e)
                    color_profile = profile_list.get_default_video_stream_profile()
                    print("color profile: ", color_profile)
                config.enable_stream(color_profile)
            except Exception as e:
                print(e)
                return
            pipeline.start(config)

            def read_orbbec_frame():
                frames: FrameSet = pipeline.wait_for_frames(1000)
                if frames is None:
                    return False, None
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    return False, None
                # covert to RGB format
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    print("failed to convert frame to image")
                    return False, None
                return True, color_image

            self._video_read = read_orbbec_frame
            self._frame_size = (640, 480)
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

        # print(f"Frame {self.frame}")
        self.frame += 1
        self._stream.write(frame)

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
