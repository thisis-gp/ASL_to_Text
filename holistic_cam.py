import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

ROWS_PER_FRAME = 543  # number of landmarks per frame

# Function to predict sign based on parquet
def get_prediction(prediction_fn, pq_file):
    xyz_np = load_relevant_data_subset(pq_file)
    prediction = prediction_fn(inputs=xyz_np)
    pred = prediction['outputs'].argmax()
    sign = ORD2SIGN[pred]
    pred_conf = prediction['outputs'][pred]
    return sign, pred_conf

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def create_frame_landmarks_df(results, frame, xyz):
    """
    Takes results from mediapipe and creates a dataframe of landmarks
    inputs:
        results: mediapipe results object
        frame: frame number
        xyz: dataframe of xyz example data
    """
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()
    if results.face_landmarks:
        for i,point in enumerate(results.face_landmarks.landmark):
            face.loc[i,["x", "y", "z"]] = [point.x, point.y, point.z]


    if results.pose_landmarks:
        for i,point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i,["x", "y", "z"]] = [point.x, point.y, point.z]


    if results.left_hand_landmarks:
        for i,point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i,["x", "y", "z"]] = [point.x, point.y, point.z]


    if results.right_hand_landmarks:
        for i,point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i,["x", "y", "z"]] = [point.x, point.y, point.z]
    
    face = face.reset_index().rename(columns={'index':'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index':'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='right_hand')
    landmarks = pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type','landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

ex_pq_file = "sample.parquet"
xyz = pd.read_parquet(ex_pq_file)

train = pd.read_csv("train.csv")

interpreter = tf.lite.Interpreter("model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")
# Add ordinally Encoded Sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes
# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

pq_file = "output.parquet"

cap = cv2.VideoCapture(0)

# Set the fullscreen window
cv2.namedWindow("Holistic Model Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Holistic Model Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_no = 0
    while cap.isOpened():
        frame_no += 1
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_rgb.flags.writeable = False
        results = holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Extracting landmark data for prediction
        if results.left_hand_landmarks or results.right_hand_landmarks:
            landmarks_df = create_frame_landmarks_df(results, frame_no, xyz)
            landmarks_df.to_parquet('output.parquet')
            # Gather landmarks from both hands
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

            # Bounding box calculation for hand landmarks
            if x_ and y_:
                x1 = int(min(x_) * W)
                y1 = int(min(y_) * H)
                x2 = int(max(x_) * W)
                y2 = int(max(y_) * H)

                sign, pred_conf = get_prediction(prediction_fn, pq_file)

                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 99, 173), 2)
                cv2.putText(frame_rgb, f"{sign} ({pred_conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Holistic Model Detection', frame_rgb)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()