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

import argparse
import asyncio
import json
import logging
import os
import platform
import ssl
import queue
from typing import Optional
import threading

from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    VideoStreamTrack,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaPlayer, MediaRelay

ROOT = os.path.dirname(__file__)

# def create_local_tracks(play_from):
#     player = MediaPlayer(play_from)
#     return player.video


async def index(request: web.Request) -> web.Response:
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request: web.Request) -> web.Response:
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request, vqueue):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    # open media source
    # video = create_local_tracks("video.mp4")

    pc.addTrack(vqueue)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


class MyVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue(10)

    async def recv(self):
        img = self.queue.get()
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


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

        self._vqueue = MyVideoTrack()
        self.app = web.Application()
        self.app.on_shutdown.append(on_shutdown)
        self.app.router.add_get("/", index)
        self.app.router.add_get("/client.js", javascript)
        self.app.router.add_post("/offer", lambda x: offer(x, self._vqueue))
        t = threading.Thread(target=lambda: web.run_app(self.app, host="0.0.0.0", port=8080), daemon=True)
        t.start()

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
            # cv2.imshow("drone_feed", image)
            # cv2.waitKey(1)
            # cv2.waitKey(0)
            try:
                self._vqueue.queue.put(image, block=False)
            except:
                pass
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
