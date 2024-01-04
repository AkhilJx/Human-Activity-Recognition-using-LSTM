# Human-Activity-Recognition-using-LSTM
Real-Time Human Activity Recognition Using LSTM

Introduction:
This repository contains a Python script for real-time human activity recognition using pose estimation. it is divided into 2 parts: one is the model building part where we build a model for activity recognition using LSTM. Next part is real-time testing in which we load the model and test on real data. It utilizes the MediaPipe library for holistic pose estimation and an LSTM (Long Short-Term Memory) neural network for activity classification. The program captures video input from a webcam, detects human poses, and predicts the performed activity.

Libraries and Dependencies:
The code relies on the following libraries:

OpenCV: For capturing and processing video frames.
NumPy: Used for numerical operations and array manipulation.
Matplotlib: Employed for visualization purposes.
Mediapipe: Utilized for holistic pose estimation.
TensorFlow and Keras: Used for building and training the LSTM neural network.
Pose Estimation and Feature Extraction:
The script uses the MediaPipe library to perform holistic pose estimation. The detected pose landmarks are then processed to extract relevant features for activity recognition. These features include angles and distances between key body parts.

LSTM Neural Network:
The LSTM neural network is implemented using the TensorFlow and Keras frameworks. It consists of multiple layers of LSTM units followed by dense layers. The model is trained on a dataset containing sequences of pose features corresponding to different activities.

Real-time Activity Recognition:
The program continuously captures video frames from the webcam, performs pose estimation, extracts features, and feeds them into the trained LSTM model for activity prediction. The recognized activity is displayed on the screen in real-time.

Instructions for Use:
To use the code, ensure you have the required dependencies installed. You can run the script by executing the provided Python file. The webcam feed will be displayed with real-time activity recognition results.

Additional Information:
The code supports the recognition of activities such as standing, sitting, and kneeling. You can customize the activities by modifying the 'actions' array.
The trained model weights are loaded from the 'action2.h5' file.
Press 'Q' to quit the application.
Future Improvements:
The repository can be extended by incorporating additional activities for recognition, enhancing the dataset for training, and optimizing the model architecture for better performance.

Feel free to explore, modify, and contribute to this project. If you encounter issues or have suggestions, please open an issue on GitHub.

Sample Output:
![image](https://github.com/AkhilJx/Human-Activity-Recognition-using-LSTM/assets/78065413/88b824bc-a4f7-4bea-8d14-5d44a4a3353d)

