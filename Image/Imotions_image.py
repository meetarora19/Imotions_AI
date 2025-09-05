import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import tempfile
import gradio as gr

base_dir = "replace this with the folder path that contains the dataset"
# https://www.kaggle.com/datasets/msambare/fer2013

def load_images_and_labels(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    for label in class_names:
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels), class_names

# Load dataset
X_train, y_train, class_names = load_images_and_labels(os.path.join(base_dir, "train"))
X_test, y_test, _ = load_images_and_labels(os.path.join(base_dir, "test"))

X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
X_test = X_test.reshape(-1, 48, 48, 1) / 255.0

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

y_train_cat = to_categorical(y_train_enc, num_classes=len(class_names))
y_test_cat = to_categorical(y_test_enc, num_classes=len(class_names))

# Load or train model
if os.path.exists("best_emotion_cnn_model.h5"):
    print("Loading best saved model...")
    model = load_model("best_emotion_cnn_model.h5")
elif os.path.exists("final_emotion_cnn_model.h5"):
    print("Loading final saved model...")
    model = load_model("final_emotion_cnn_model.h5")
else:
    print("No saved model found. Training from scratch...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_emotion_cnn_model.h5', save_best_only=True, monitor='val_loss')

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop, checkpoint]
    )

    model.save('final_emotion_cnn_model.h5')

    # Evaluate & show report
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Training history plots
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# ==================== GRADIO UI ====================

def predict_emotions(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_cv, (48, 48))
    img_norm = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_norm, axis=(0, -1))

    try:
        pred_probs = model.predict(img_input)
        pred_class = np.argmax(pred_probs)
        pred_emotion = encoder.inverse_transform([pred_class])[0]
        cnn_probs = {label: float(prob * 100) for label, prob in zip(encoder.classes_, pred_probs[0])}
    except Exception:
        pred_emotion = "Error"
        cnn_probs = {}

    output_str = (
        f"<div style='margin:0; padding:0; font-size:1.5rem; font-weight:bold;'>"
        f"<span style='color:#87CEEB;'>Predicted Emotion:</span> "
        f"<span style='color:white;'>{pred_emotion.capitalize()}</span>"
        f"</div>"
    )

    if cnn_probs:
        plt.figure(figsize=(6, 3))
        plt.barh(list(cnn_probs.keys()), list(cnn_probs.values()), color='skyblue')
        plt.xlim(0, 100)
        plt.xlabel("Confidence (%)")
        plt.title("CNN Emotion Probabilities")
        plt.tight_layout()
        cnn_prob_chart = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(cnn_prob_chart.name)
        plt.close()
        cnn_prob_chart_path = cnn_prob_chart.name
    else:
        cnn_prob_chart_path = None

    return output_str, cnn_prob_chart_path

def create_placeholder_image():
    bg_color = (31, 41, 55)
    size = (450, 250)
    img = Image.new('RGBA', size, bg_color)
    d = ImageDraw.Draw(img)
    main_text, sub_text = "Upload a file", "to check its emotion confidence"
    try:
        main_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 80)
    except:
        main_font = ImageFont.load_default()
    try:
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 46)
    except:
        sub_font = ImageFont.load_default()

    main_bbox = d.textbbox((0, 0), main_text, font=main_font)
    main_w, main_h = main_bbox[2] - main_bbox[0], main_bbox[3] - main_bbox[1]
    sub_bbox = d.textbbox((0, 0), sub_text, font=sub_font)
    sub_w, sub_h = sub_bbox[2] - sub_bbox[0], sub_bbox[3] - sub_bbox[1]
    total_h = main_h + sub_h + 60
    y0 = (size[1] - total_h) // 2

    d.text(((size[0] - main_w) // 2, y0), main_text, font=main_font, fill=(255, 255, 255, 255))
    d.text(((size[0] - sub_w) // 2, y0 + main_h + 60), sub_text, font=sub_font, fill=(180, 192, 210, 255))

    placeholder_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    img.save(placeholder_path)
    return placeholder_path

placeholder_image_path = create_placeholder_image()

css_code = """
.fixed-image {
    width: 450px !important;
    height: 250px !important;
    max-width: 450px !important;
    max-height: 250px !important;
    overflow: hidden;
    aspect-ratio: 1.8/1 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #1F2937;
    border-radius: 6px;
}
.fixed-image img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    cursor: zoom-in;
    border-radius: 6px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="red"), css=css_code) as demo:
    gr.Markdown("<div style='text-align:center; font-size:3rem; font-weight:bold; color:#87CEEB;'>Imotions</div>")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", show_label=False, interactive=True, elem_classes="fixed-image")
            with gr.Row():
                clear = gr.Button("Clear")
                submit = gr.Button("Predict")
        with gr.Column():
            output_text = gr.Markdown(show_label=False)
            output_chart = gr.Image(value=placeholder_image_path, show_label=False, interactive=False, elem_classes="fixed-image")

    def predict_and_display(image):
        output_str, cnn_prob_chart_path = predict_emotions(image)
        chart = cnn_prob_chart_path or placeholder_image_path
        return output_str, chart

    def clear_all():
        return None, None, placeholder_image_path

    submit.click(fn=predict_and_display, inputs=input_img, outputs=[output_text, output_chart])
    clear.click(fn=clear_all, inputs=[], outputs=[input_img, output_text, output_chart])

demo.launch(share=True)
