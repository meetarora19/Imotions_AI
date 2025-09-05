# Imotions (Emotion Detector Using Images)

## Overview

This implements a convolutional neural network (CNN) to recognize emotions from facial images. It includes:

- Data loading and preprocessing of grayscale facial images.
- Training a CNN model to classify emotions.
- Saving the best and final trained models (`best_emotion_cnn_model.h5` and `final_emotion_cnn_model.h5`).
- A Gradio-based web UI for real-time emotion prediction from images.


## Usage

### Training the model separately (optional)

1. Place your training images in the `train` folder and testing images in the `test` folder, organized by emotion classes (subfolders named by emotion).
2. Run the `model_training.py` script to start training the CNN model.
3. The trained models will be saved as `best_emotion_cnn_model.h5` and `final_emotion_cnn_model.h5`.
4. Evaluation results and training loss/accuracy plots will be displayed.


### Running the main emotion recognition app

- The `imotions_image.py` script attempts to load existing model files first.
- If models are not found, it will automatically train a new model.
- After loading/training, the Gradio UI will launch for live emotion prediction from uploaded images.


## Notes

- Images are resized to 48x48 pixels and converted to grayscale.
- CNN architecture includes Conv2D, MaxPooling, Dropout, Dense layers.
- EarlyStopping and ModelCheckpoint callbacks optimize training.
- Training history plots and confusion matrix are displayed for analysis.
