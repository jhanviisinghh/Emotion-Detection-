# Emotion Detection Model

👁️‍🗨️ This is an emotion detection model trained on the [Facial Expression Recognition Challenge dataset](https://www.kaggle.com/debanga/facial-expression-recognition-challenge).

## Dataset
📂 **Dataset**: Download the dataset from [Kaggle](https://www.kaggle.com/debanga/facial-expression-recognition-challenge).
   After downloading, place the dataset CSV file (`icml_face_data.csv`) in the `dataset` directory.

## Model Training

### Requirements
- Python 3
- TensorFlow
- NumPy
- pandas
- Matplotlib
- scikit-learn

### Steps
1. Run `train_emotion_model.py` to train the emotion detection model.
2. The model will be saved as `model.h5`.

## Usage

### Detect Emotions from Webcam

1. Install required libraries:
2. Run `detect_emotion.py` to start detecting emotions from your webcam.

## File Descriptions

- `train_emotion_model.py`: Script to train the emotion detection model.
- `detect_emotion.py`: Script to detect emotions from a webcam feed.
- `haarcascade_frontalface_default.xml`: Haar cascade file for face detection.

## Emotion Labels
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Feel free to explore and modify the code according to your needs! 😊
