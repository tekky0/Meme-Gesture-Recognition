import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import csv



hand_mp = mp.solutions.hands
face_mp = mp.solutions.face_mesh

hands =hand_mp.Hands(max_num_hands=2, min_detection_confidence=0.7);
face = face_mp.FaceMesh(min_detection_confidence=0.7);
mp_draw = mp.solutions.drawing_utils

match input("mode (def|log|pro):"):
    case "def":
        cap = cv.VideoCapture(0);
        while True:

            num, frame = cap.read();
            y,x,c = np.shape(frame);
            frame = cv.flip(frame, 1);
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB);
            result_hands = hands.process(rgb);
            result_face = face.process(rgb);
            
            
            if result_hands.multi_hand_landmarks:
                for i, handslms in enumerate(result_hands.multi_hand_landmarks):
                    landmarks = [];     
                    for lm in handslms.landmark:
                        
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])
                        
                    int_minx = int(min(l[0] for l in landmarks))-20;
                    int_miny = int(min(l[1] for l in landmarks))-20;
                    int_maxx = int(max(l[0] for l in landmarks))+20;
                    int_maxy = int(max(l[1] for l in landmarks))+20;
                    text = str(result_hands.multi_handedness[i].classification[0].label);
                    cv.rectangle(frame, (int_minx,int_miny),(int_maxx,int_maxy), (255,255,255), 2);
                    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 1)[0];
                    cv.rectangle(frame, (int_minx, int_miny), (int_minx + text_size[0], int_miny - text_size[1]), (0,0,0), -1);
                    cv.putText(frame, str(result_hands.multi_handedness[i].classification[0].label), (int_minx, int_miny),cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

                    mp_draw.draw_landmarks(frame, handslms, hand_mp.HAND_CONNECTIONS)
            
            if(result_face.multi_face_landmarks):
                for face_landmarks in result_face.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, face_landmarks, face_mp.FACEMESH_LIPS);
            cv.imshow('video', frame);
            if cv.waitKey(1) == 27:
                break;
    case "log":
        captureIsTrue = 0;
        with open("D:\\.vscode\\.vscode\\main_projects\\venv\\poseData.csv", 'a'):
            read = csv.reader
        cap = cv.VideoCapture(0);
        label = input("input sample name: ");
        # isHandActive, isFaceActive = input("detect Hand and or Face\n(set 1 per option ' ' == delimiter): ").split(' ');
        # isHandActive = int(isHandActive);
        # isFaceActive = int(isFaceActive);
        isHandActive = 1;
        isFaceActive = 1;
        maxSampleSize = 500;
        sampleCount = 0;
        
        leftHand = np.zeros(42);
        rightHand = np.zeros(42);
        
        writer = csv.writer(open("D:\\.vscode\\.vscode\\main_projects\\venv\\poseData.csv", "a", newline=''));        
        
        while True:
            
            capture = False;
            

            num, frame = cap.read();
            y,x,c = np.shape(frame);
            frame = cv.flip(frame, 1);
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB);
            result_hands = hands.process(rgb);
            result_face = face.process(rgb);
            canvas = np.zeros((y,2*x,3), dtype=np.uint8);
            cv.line(canvas,(x,0), (x,y), (255,255,255), 1);
            textH = cv.getTextSize("Data Log", cv.FONT_HERSHEY_SIMPLEX, 3, 2)[0][1]
            cv.putText(canvas, "Data Log", (x+5, 5+textH), cv.FONT_HERSHEY_SIMPLEX, 3,(255,255,255), 2);
            cv.putText(canvas, f"Sample: {sampleCount}", (x+5, 10+textH), cv.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1);
            
            
            if(isHandActive):
                if result_hands.multi_hand_landmarks:
                    for i, handslms in enumerate(result_hands.multi_hand_landmarks):
                        base_x, base_y = handslms.landmark[0].x, handslms.landmark[0].y
                        landmarks = [];     
                        coords = [];
                        for lm in handslms.landmark:
                            coords.append(lm.x - base_x)
                            coords.append(lm.y - base_y)
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)
                            landmarks.append([lmx, lmy])
                        max_val = max(abs(v) for v in coords) or 1
                        normalized = [v / max_val for v in coords]
                        side = result_hands.multi_handedness[i].classification[0].label
                        if side == "Left":
                            rightHand = normalized
                        else:
                            leftHand = normalized
                            
                        int_minx = int(min(l[0] for l in landmarks))-20;
                        int_miny = int(min(l[1] for l in landmarks))-20;
                        int_maxx = int(max(l[0] for l in landmarks))+20;
                        int_maxy = int(max(l[1] for l in landmarks))+20;
                        text = str(result_hands.multi_handedness[i].classification[0].label);
                        cv.rectangle(frame, (int_minx,int_miny),(int_maxx,int_maxy), (255,255,255), 2);
                        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 1)[0];
                        cv.rectangle(frame, (int_minx, int_miny), (int_minx + text_size[0], int_miny - text_size[1]), (0,0,0), -1);
                        cv.putText(frame, str(result_hands.multi_handedness[i].classification[0].label), (int_minx, int_miny),cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                        mp_draw.draw_landmarks(frame, handslms, hand_mp.HAND_CONNECTIONS);
                       
                if(result_face.multi_face_landmarks and isFaceActive):
                    for face_landmarks in result_face.multi_face_landmarks:
                        mp_draw.draw_landmarks(frame, face_landmarks, face_mp.FACEMESH_LIPS);
                    lms = result_face.multi_face_landmarks[0].landmark
                    base_x, base_y, base_z = lms[1].x, lms[1].y, lms[1].z
                    FaceData = [];
                    for lm in lms:
                        FaceData.append(lm.x - base_x);
                        FaceData.append(lm.y - base_y);
                        FaceData.append(lm.z - base_z);          
                                  
            canvas[:,:x] = frame;
            key = cv.waitKey(1) & 0xFF; 
            cv.imshow('video', canvas);
            if key == 27:
                break;
            elif key == ord(' '):
                fd = FaceData if isinstance(FaceData, list) else FaceData.tolist()
                row = [label] + fd + list(leftHand) + list(rightHand)
                writer.writerow(row);
                sampleCount += 1
                print(f"saved {sampleCount}/{maxSampleSize}")
            elif (key == ord('n')):
                sampleCount = 0;
                import time as t
                print("recording in 3...");
                t.sleep(1);
                print("recording in 2..");
                t.sleep(1);
                print("recording in 1.");
                t.sleep(1);
                print("running..");
                captureIsTrue = 1;
            elif(captureIsTrue):
                if(sampleCount >= maxSampleSize):
                    captureIsTrue = 0;
                fd = FaceData if isinstance(FaceData, list) else FaceData.tolist()
                row = [label] + fd + list(leftHand) + list(rightHand)
                writer.writerow(row);
                sampleCount += 1
                print(f"saved {sampleCount}/{maxSampleSize}")
                
                
                
    case "pro":
        retrain = input("re-train sequential model? (y/n): ")
        if retrain == 'y':
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder

            print("re-training...")
            df = pd.read_csv("D:\\.vscode\\.vscode\\main_projects\\venv\\poseData.csv", header=None)
            X = df.iloc[:, 1:].values
            y = df.iloc[:, 0].values

            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            np.save("labels.npy", le.classes_)
            print(f"Classes: {le.classes_}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42)

            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(len(le.classes_), activation='softmax')
            ])
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_test, y_test), verbose=1)
            model.save("poseModel")
            print("Model saved!")
        else:
            model = tf.keras.models.load_model("poseModel")

        labels = np.load("labels.npy", allow_pickle=True)
        print("model loaded.")
        print("importing meme assets...")
        dict = {
            # "uhhhh....": cv.imread(),
            
                "kissy":cv.imread("venv\\images\\kissy_hamster.jpg"),
                 "nerd":cv.imread("venv\\images\\nerd_hamster.jpg"),
            "thumbs_up":cv.imread("venv\\images\\thumbs_up_hamster.jpg"),
               "peace": cv.imread("venv\\images\\peace_hamster.jpg"),
         "thumbs_down": cv.imread("venv\\images\\thumbs_down_hamster.jpg"),
       "pretty_please": cv.imread("venv\\images\\pretty_please_hamster.jpg"),
           "love_you" : cv.imread("venv\\images\\love_you_hamster.gif"),
          "uhhhh...." : cv.imread("venv\\images\\monkey_thinking.jpg"),
           "fuck_you" : cv.imread("venv\\images\\fuck_you_hamster.jpg")
        }

        cap = cv.VideoCapture(0)
        leftHand = np.zeros(42)
        rightHand = np.zeros(42)
        faceData = np.zeros(1404)

        while True:
            num, frame = cap.read()
            y, x, c = np.shape(frame)
            frame = cv.flip(frame, 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result_hands = hands.process(rgb)
            result_face = face.process(rgb)
            canvas = np.zeros((y, 2*x, 3), dtype=np.uint8)

            # reset each frame
            leftHand = np.zeros(42)
            rightHand = np.zeros(42)
            faceData = np.zeros(1404)

            if result_hands.multi_hand_landmarks:
                for i, handslms in enumerate(result_hands.multi_hand_landmarks):
                    base_x, base_y = handslms.landmark[0].x, handslms.landmark[0].y
                    landmarks = []
                    coords = []
                    for lm in handslms.landmark:
                        coords.append(lm.x - base_x)
                        coords.append(lm.y - base_y)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])

                    max_val = max(abs(v) for v in coords) or 1
                    normalized = [v / max_val for v in coords]
                    side = result_hands.multi_handedness[i].classification[0].label
                    if side == "Left":
                        rightHand = normalized
                    else:
                        leftHand = normalized

                    int_minx = min(l[0] for l in landmarks) - 20
                    int_miny = min(l[1] for l in landmarks) - 20
                    int_maxx = max(l[0] for l in landmarks) + 20
                    int_maxy = max(l[1] for l in landmarks) + 20
                    text = side
                    cv.rectangle(frame, (int_minx, int_miny), (int_maxx, int_maxy), (255,255,255), 2)
                    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                    cv.rectangle(frame, (int_minx, int_miny - text_size[1] - 5),
                                (int_minx + text_size[0], int_miny), (0,0,0), -1)
                    cv.putText(frame, text, (int_minx, int_miny - 5),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                    mp_draw.draw_landmarks(frame, handslms, hand_mp.HAND_CONNECTIONS)

            if result_face.multi_face_landmarks:
                for face_landmarks in result_face.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, face_landmarks, face_mp.FACEMESH_LIPS,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_draw.DrawingSpec(
                                            color=(0,255,0), thickness=1))
                lms = result_face.multi_face_landmarks[0].landmark
                base_x, base_y, base_z = lms[1].x, lms[1].y, lms[1].z
                faceData = []
                for lm in lms:
                    faceData.append(lm.x - base_x)
                    faceData.append(lm.y - base_y)
                    faceData.append(lm.z - base_z)

            # predict
            fd = faceData if isinstance(faceData, list) else faceData.tolist()
            input_data = np.array([fd + list(leftHand) + list(rightHand)])
            pred = model.predict(input_data, verbose=0)
            confidence = np.max(pred) * 100
            pose = labels[np.argmax(pred)]

            canvas[:, :x] = frame
            # draw prediction on canvas menu side
            cv.line(canvas, (x, 0), (x, y), (255,255,255), 1)
            if confidence > 80:
                color = (0, 255, 0)
            if pose in dict:
                img_h, img_w = y,x
                img = cv.resize(dict[pose], (img_w, img_h))
                canvas[:, x:] = img
            else:
                color = (100, 100, 100)
            cv.putText(canvas, f"{pose}", (x+10, 60),
                    cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv.putText(canvas, f"{confidence:.0f}%", (x+10, 100),
                    cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv.imshow('video', canvas)
            key = cv.waitKey(1)
            if key == 27:
                break
                    
            
cap.release();
cv.destroyAllWindows();
    