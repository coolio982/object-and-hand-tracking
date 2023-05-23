# Hand Gesture Recognition
Repository for training hand gesture data, based off of mediapipe.<br> 
<br> ❗ Based off https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe  ❗
<br> 




# Running the program
```bash
python main.py
```

# Directory
<pre>
│  main.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>
### main.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is a jupyter notebook for training a model forsign recognition.

### point_history_classification.ipynb
This is a jupyter notebook for training a model for gesture recognition.

### model/keypoint_classifier
The following files are found in this directory:
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### model/point_history_classifier
The following files are found in this directory:
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv)
* Inference module(point_history_classifier.py)

### utils/cvfpscalc.py
This file contains fps measurement calculations.

### utils/gesturecalcs.py
This file contains calculations for bounding boxes and displays regarding the gestures. 

# Training
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

## Sign recognition
### 1. Data Colleciton
Press k to enter the mode to start saving keypoints to  "model/keypoint_classifier/keypoint.csv"<br>
0-9 can be used as labels for the keypoints. Modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.

### 2. Model training
"[keypoint_classification.ipynb](keypoint_classification.ipynb)" contains the full training and inference procedure. Hyperparameter tuning is also included; you may change these as appropriate. <br>

## Gesture recognition 
### 1. Data Collection
Press h to enter the mode to save the history of fingertip coordinates to "model/point_history_classifier/point_history.csv". Currently the location of the index finger is tracked.<br>
0-9 can be used as labels for the keypoints. Modify the label of "model/point_history_classifier/point_history_classifier_label.csv" as appropriate.<br>


### 2. Model training
"[point_history_classification.ipynb](point_history_classification.ipynb)" contains the full training and inference procedure. Note that LSTM network is optional, but preferred.




# Reference
* [MediaPipe](https://mediapipe.dev/)


