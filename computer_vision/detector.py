import sys
import cv2


def main():
    print("Acquiring video")
    video = cv2.VideoCapture(
        "udpsrc port=5600 ! application/x-rtp,payload=96,encoding-name=H264 ! rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink",
        cv2.CAP_GSTREAMER,
    )
    if not video.isOpened():
        print("Failed to open")
        sys.exit()

    while True:
        ret, frame = video.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Display the frame
        cv2.imshow("drone_feed", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
