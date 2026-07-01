import cv2 as cv
import mediapipe as mp
import csv
import os

mp_hand = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hand.Hands(max_num_hands=1)
face = mp_face.FaceMesh(max_num_faces=1)

cap = cv.VideoCapture(0)

label = input("Enter gesture label: ")
save = False
count = 0
target = 300

csv_path = "gesture_data.csv"
file_exists = os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        hand_cols = [f"h_{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        face_cols = [f"f_{axis}{i}" for i in range(468) for axis in ["x", "y", "z"]]
        writer.writerow(["label"] + hand_cols + face_cols)

    while True:
        c_suc, frame = cap.read()
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        hand_res = hands.process(rgb_frame)
        face_res = face.process(rgb_frame)

        hand_data = []
        face_data = []

        if hand_res.multi_hand_landmarks:
            for hand_landmarks in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    hand_data += [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]

        if face_res.multi_face_landmarks:
            for face_landmarks in face_res.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
                nose = face_landmarks.landmark[1]  # nose tip as reference point
                for lm in face_landmarks.landmark:
                    face_data += [lm.x - nose.x, lm.y - nose.y, lm.z - nose.z]

        # only save if both hand and face detected
        # if save and len(hand_data) == 63 and len(face_data) == 1404:
            writer.writerow([label] + hand_data + face_data)
            count += 1

        status = f"RECORDING {count}/{target}" if save else "Press SPACE to record"
        cv.putText(frame, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if save else (0, 0, 255), 2)
        cv.imshow("frame", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord(' '):
            save = not save
        if key == ord('q') or count >= target:
            break

cap.release()
cv.destroyAllWindows()
print(f"Saved {count} samples for '{label}'")