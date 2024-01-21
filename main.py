import cv2

from emotion_detection import EmotionDetector


def get_top_emotion(emotions):
    for face in emotions:
        face["top_emotion"] = max(face["emotions"], key=face["emotions"].get)
    return emotions


def webcam(device):
    detector = EmotionDetector()
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)

        emotions = get_top_emotion(detector.detect_emotions(frame))
        if len(emotions):
            for face in emotions:
                x, y, w, h = face["box"]
                emotions = face["emotions"]
                cv2.rectangle(
                    frame,
                    (x, y, w, h),
                    (0, 155, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    face["top_emotion"],
                    (
                        x,
                        y + h + 15 + 15,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 155, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam(0)
