import sys
import time
import os

import cv2
# from aiortc import VideoStreamTrack

from ultralytics import YOLO
import torch

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude

import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling
from av import VideoFrame
import queue
import fractions
from datetime import datetime


class CustomVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        # self.cap = cv2.VideoCapture(camera_id)
        self.queue = queue.Queue(10)
        self.frame_count = 0

    async def recv(self):
        self.frame_count += 1
        print(f"Sending frame {self.frame_count}")
        try:
            frame = self.queue.get(block=False)
        except queue.Empty:
            return None
        # ret, frame = self.cap.read()
        # if not ret:
        #     print("Failed to read frame from camera")
        #     return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)  # Use fractions for time_base
        # Add timestamp to the frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Current time with milliseconds
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)  # Use fractions for time_base
        return video_frame


async def setup_webrtc_and_run(ip_address, port, track):
    signaling = TcpSocketSignaling(ip_address, port)
    pc = RTCPeerConnection()
    # video_sender = CustomVideoStreamTrack(camera_id)
    video_sender = track
    pc.addTrack(video_sender)

    try:
        await signaling.connect()

        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"Data channel established: {channel.label}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "connected":
                print("WebRTC connection established successfully")

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await signaling.send(pc.localDescription)

        while True:
            obj = await signaling.receive()
            if isinstance(obj, RTCSessionDescription):
                await pc.setRemoteDescription(obj)
                print("Remote description set")
            elif obj is None:
                print("Signaling ended")
                break
        print("Closing connection")
    finally:
        await pc.close()


async def runmain(track):
    ip_address = "localhost"  # Ip Address of Remote Server/Machine
    port = 9999
    # camera_id = 2  # Change this to the appropriate camera ID
    await setup_webrtc_and_run(ip_address, port, track)


class Detector(Node):
    def __init__(self):
        super().__init__("Detector")
        self._position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._position_cb,
            10,
        )
        self._position = VehicleLocalPosition()

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
        if not self._video.isOpened():
            print("Failed to open video")
            rclpy.shutdown()

        # asyncio.run(main())
        self._stream = CustomVideoStreamTrack()
        runmain(self._stream)
        self.periodic = 0
        self.create_timer(0.05, self._detect)

    def _position_cb(self, msg: VehicleLocalPosition):
        self._position = msg

    def _detect(self):
        ret, frame = self._video.read()

        if not ret:
            print("Failed to grab frame.")
            return

        results = self._model.predict(frame, stream=True)
        for r in results:
            image = r.plot()
            original = r.orig_shape
            cv2.line(
                image,
                (r.orig_shape[1] // 2, 0),
                (r.orig_shape[1] // 2, r.orig_shape[0]),
                (0, 0, 0),
                1,
            )
            cv2.line(
                image,
                (0, r.orig_shape[0] // 2),
                (
                    r.orig_shape[1],
                    r.orig_shape[0] // 2,
                ),
                (0, 0, 0),
                1,
            )
            try:
                self._stream.queue.put(image, block=False)
            except queue.Full:
                pass
            # cv2.imshow("drone_feed", image)
            # cv2.waitKey(1)
            # cv2.waitKey(0)
            if len(r.boxes) > 0:
                box = list(r.boxes[0].xyxy[0].to("cpu"))
                print(box)
                print(original)
                center = original[1] / 2, original[0] / 2
                centroid = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                x = centroid[0] - center[0]  # NED body frame
                y = center[1] - centroid[1]  # NED body frame

    def shut_down_cb(self):
        self._video.release()


def main(args=None):
    rclpy.init(args=args)
    detector = Detector()
    rclpy.spin(detector)

    detector.shut_down_cb()
    cv2.destroyAllWindows()
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
