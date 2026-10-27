
import cv2
import mediapipe as mp
import csv
import os

FILE = "dataset.csv"

gestures = {
    ord("1"): 0,  # Heart
    ord("2"): 1,  # Nerd
    ord("3"): 2,  # ThumbsUp
    ord("4"): 3,  # ThumbsDown
    ord("5"): 4,  # Freak
}

names = ["Heart", "Nerd", "ThumbsUp", "ThumbsDown", "Freak"]

hands = mp.solutions.hands.Hands(max_num_hands=1)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

try:
    with open(FILE, "a", newline="") as file:
        writer = csv.writer(file)

        print("Press 1-5 to record a gesture.")
        print("1 Heart | 2 Nerd | 3 ThumbsUp")
        print("4 ThumbsDown | 5 Freak")
        print("Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            message = "Show your hand | Press 1-5"

            if results.multi_hand_landmarks:
                landmark_set = results.multi_hand_landmarks[0]
                data = []

                for point in landmark_set.landmark:
                    data.extend([point.x, point.y])

                message = "Press 1-5 to save this gesture"

                cv2.putText(
                    frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

                key = cv2.waitKey(1) & 0xFF

                if key in gestures:
                    label = gestures[key]
                    writer.writerow(data + [label])
                    file.flush()
                    print(f"Saved {names[label]}")

            else:
                cv2.putText(
                    frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )
                key = cv2.waitKey(1) & 0xFF

            cv2.imshow("Collect Gesture Data", frame)

            if key == ord("q"):
                break

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()
