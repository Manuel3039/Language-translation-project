# Multilingual Translator

A simple web app for translating text between 15 selected languages using
Facebook's pretrained [M2M100](https://huggingface.co/facebook/m2m100_418M)
translation model and Gradio.

## Features

- Translate text between English, French, Spanish, German, Chinese
  (Simplified), Hindi, Arabic, Russian, Portuguese, Japanese, Korean, Italian,
  Dutch, Turkish, and Polish.
- Choose source and target languages in a browser-based interface.
- Run locally on CPU or with a supported PyTorch accelerator.

## Requirements

- Python 3.9 or newer
- Internet access on first launch to download the model and tokenizer

## Setup

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed in the terminal (typically
`http://127.0.0.1:7860`). The model is downloaded from Hugging Face the first
time the app starts, so initial startup may take a while.

## Notes

- The M2M100 model supports many languages; this app currently presents the
  15 languages listed above.
- CPU translation can be slow. PyTorch may use an available supported
  accelerator automatically.
- The app launches locally by default. Avoid enabling Gradio public sharing
  for sensitive text.

## Model

This app uses `facebook/m2m100_418M`, a multilingual sequence-to-sequence
translation model hosted on Hugging Face.
