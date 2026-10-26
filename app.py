
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

# Load the gesture recognition model
model = nn.Sequential(
    nn.Linear(42, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 5)
)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Gesture names
labels = ["Heart", "Nerd", "ThumbsUp", "ThumbsDown", "Freak"]

# Start hand detection
hands = mp.solutions.hands.Hands(max_num_hands=1)
camera = cv2.VideoCapture(0)

while camera.isOpened():
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    text = "No hand detected"

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]

        data = []
        for point in landmarks.landmark:
            data.extend([point.x, point.y])

        with torch.no_grad():
            prediction = model(torch.tensor(data).float().unsqueeze(0))

        gesture = torch.argmax(prediction).item()
        text = labels[gesture]

    cv2.putText(
        frame, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 0), 2
    )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
hands.close()
cv2.destroyAllWindows()
