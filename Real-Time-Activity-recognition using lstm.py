# 1. Import necessary libraries
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# 2. Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# function for mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# function for drawing landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections


def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )


# function for calculating the necessary angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# function for calculating the necessary distances
def calculate_dist(a,b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)
    return dist


def n1(x):
    return ((x-0)/(180-0))


def n2(x):
    return ((x-0)/(33-0))


# function to return necessary parameters
def get_coordinates(l):
    landmarks = l

    # Calculate Coordinates for left parameters
    lshoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
    lhip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
    lankle = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x,
              landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y]
    lknee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,
             landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
    lfootidx = [landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    #     lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    #     lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # Calculate Coordinates for right parameters
    rshoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
    rhip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
    rankle = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x,
              landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y]
    rknee = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x,
             landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
    rfootidx = [landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    #     relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    #     rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # Calculate required PARAMETERS
    rhangle = calculate_angle(rshoulder, rhip, rknee)
    lhangle = calculate_angle(lshoulder, lhip, lknee)
    rkangle = calculate_angle(rankle, rknee, rhip)
    lkangle = calculate_angle(lankle, lknee, lhip)
    lfangle = calculate_angle(lknee, lankle, lfootidx)
    rfangle = calculate_angle(rknee, rankle, rfootidx)
    ldist = calculate_dist(lhip, lankle)
    rdist = calculate_dist(rhip, rankle)
    #     langle = calculate_angle(lshoulder, lelbow, lwrist)
    #     rangle = calculate_angle(rshoulder, relbow, rwrist)
    #     lsangle=calculate_angle(lhip,lshoulder,lelbow)
    #     rsangle=calculate_angle(rhip,rshoulder,relbow)
    #     ankdist=calculate_dist(lankle,rankle)
    #     rwdist=calculate_dist(rhip,rwrist)
    #     lwdist=calculate_dist(lhip,lwrist)

    return rhangle, lhangle, rkangle, lkangle, lfangle, rfangle, ldist, rdist


def extract_keypoints(results):
    test=[]
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(141)
    if results.pose_landmarks:
        a,b,c,d,e,f,g,h = get_coordinates(results.pose_landmarks.landmark)
        test=np.array([n1(a),n1(b),n1(c),n1(d),n1(e),n1(f),g,h,n2(len(results.pose_landmarks.landmark))])
        q=np.concatenate([pose,test])
        q=q.flatten()
        return q

    else:
        return pose


# 4. Setup Folders for Collection
# Path for exported data, numpy arrays
# DATA_PATH = os.path.join('C:/Users/Dell/Desktop/trial/')

# Actions that we try to detect
actions = np.array(['standing', 'sitting', 'kneeling'])

# Thirty videos worth of data
no_sequences = 100

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0

# 6. Preprocess Data and Create Labels and Features
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

# 7. Build and Train LSTM Neural Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,141)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('./models/action2.h5')

from scipy import stats

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(output_frame, 'Press \'q\' to Quit', (250,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        # cv2.rectangle(output_frame, (250,45), (500,80), colors[num])
    return output_frame


# 11. Test in Real Time

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #             print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # 3. Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1:
                sentence = sentence[-1:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        print("sentence is: ", sentence)
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        if len(sentence) == 0:
            cv2.putText(image, 'The person is not doing any activity', (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'The person is ' + str(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen in fullscreen
        cv2.namedWindow('Output', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Output', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()