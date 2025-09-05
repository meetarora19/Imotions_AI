# Imotions (Text Sentiment and Suicidal Ideation Analysis)

## Overview
This provides a text analysis service to detect potential suicidal ideation and determine sentiment (positive/negative) using pre-trained transformer models via Hugging Face pipelines. It offers suicide risk warnings and self-care or support suggestions based on the analysis.

## Folder Contents
- `imotions_text.py`: Main script running the suicidal ideation and sentiment classification pipelines.
- Uses datasets for validation and displays output using a Gradio web interface.

## Usage
1. Install required packages including `transformers`, `torch`, `datasets`, and `gradio`.
2. Run the script to launch a Gradio UI.
3. Input text to receive a risk assessment and a relevant suggestion or safety note.

## Notes
- Suicide risk labels: "Suicidal" or "Not Suicidal" with confidence scores.
- Sentiment analysis distinguishes positive and negative sentiments.
- Specific supportive suggestions and critical warnings are provided based on risk level.
