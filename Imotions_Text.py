import random
import torch
from transformers import pipeline
import gradio as gr
from datasets import load_dataset

# Load dataset (optional, for reference or validation)
dataset = load_dataset("Lucidest/reddit-suicidal-classify-val", split="train")
print(dataset.features)
print(dataset[0])

# Suicidal ideation classifier
suicide_classifier = pipeline(
    "text-classification",
    model="sentinet/suicidality",
    tokenizer="sentinet/suicidality",
    device=0 if torch.cuda.is_available() else -1
)

# Sentiment analysis classifier
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

positive_suggestions = [
    "Plan something exciting for the weekend 🎉",
    "Call a loved one to share your good news 📞",
    "Treat yourself to your favorite snack 🍩",
    "Write down 3 wins you've had today 🏆",
    "Dance to your favorite upbeat song 💃🕺",
    "Help someone else and spread the positivity 🤝",
    "Start a new small hobby or project 🎨",
    "Celebrate your progress no matter how small 🎊",
    "Go for a scenic walk and enjoy the moment 🌳",
    "Practice gratitude: remember 3 things you're thankful for 🙏"
]

negative_suggestions = [
    "Try a few deep breathing exercises 🌬️",
    "Take a warm shower or bath 🚿",
    "Disconnect from social media for a while 📵",
    "Write your worries in a journal ✍️",
    "Make yourself a calming cup of tea 🍵",
    "Listen to a peaceful playlist 🎶",
    "Spend quiet time with a pet 🐶🐱",
    "Step outside and focus on your senses 🌍",
    "Do a short guided meditation 🧘",
    "Reach out to a supportive friend or counselor 📞"
]

suicidal_support = [
    "🔴 You are not alone. Please find international hotlines here: https://findahelpline.com 🌍",
    "🔴 Talk to someone you trust right now — a close friend, family member, or a counselor 📞.",
    "🔴 If you're in immediate danger, please dial your local emergency number 🚨."
]

def analyze_text(text):
    try:
        label_map = {
            "LABEL_0": "Not Suicidal",
            "LABEL_1": "Suicidal",
            "suicidal": "Suicidal"  # in case model outputs raw "suicidal"
        }

        suicide_output = suicide_classifier(text)
        suicide_result = suicide_output[0]
        suicide_label_raw = suicide_result['label']
        suicide_label = label_map.get(suicide_label_raw, suicide_label_raw)
        suicide_score = round(suicide_result['score'], 3)
        print(f"Suicide Prediction → Label: {suicide_label}, Score: {suicide_score}")

        sentiment_output = sentiment_classifier(text)
        sentiment_result = sentiment_output[0]
        sentiment_label = sentiment_result['label'].upper()
        sentiment_score = round(sentiment_result['score'], 3)
        print(f"Sentiment Prediction → Label: {sentiment_label}, Score: {sentiment_score}")

        if suicide_label == "Suicidal":
            warning = (
                "🔴 CRITICAL: This text may indicate suicidal ideation. "
                "Please seek help immediately—contact a trusted person or a helpline."
            )
            return {
                "Suicide_Label": suicide_label,
                "Suicide_Confidence": suicide_score,
                "Sentiment": "CRITICAL",
                "Sentiment_Confidence": sentiment_score,
                "Safety Check": warning,
                "Support Resources": random.choice(suicidal_support),
                "icon": "🔴"
            }
        else:
            if sentiment_label == "POSITIVE":
                suggestion = random.choice(positive_suggestions)
                icon = "✅"
            else:
                suggestion = random.choice(negative_suggestions)
                icon = "⚠️"
            warning = "✅ No suicidal risk detected."
            return {
                "Suicide_Label": suicide_label,
                "Suicide_Confidence": suicide_score,
                "Sentiment": sentiment_label,
                "Sentiment_Confidence": sentiment_score,
                "Safety Check": warning,
                "Self-Care Suggestion": suggestion,
                "icon": icon
            }
    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=3, placeholder="How are you feeling today?"),
    outputs="json",
    title="Mental Health & Suicidal Ideation Detector",
    description="Detects possible suicidal ideation, provides sentiment indicator, safety note, and one supportive suggestion."
)

if __name__ == "__main__":
    demo.launch()
