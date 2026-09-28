import gradio as gr
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


MODEL_NAME = "facebook/m2m100_418M"

LANGUAGES = [
    ("English", "en"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("German", "de"),
    ("Chinese (Simplified)", "zh"),
    ("Hindi", "hi"),
    ("Arabic", "ar"),
    ("Russian", "ru"),
    ("Portuguese", "pt"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Italian", "it"),
    ("Dutch", "nl"),
    ("Turkish", "tr"),
    ("Polish", "pl"),
]


def load_model():
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()


def translate(text, source_language, target_language):
    if not text or not text.strip():
        return ""

    try:
        tokenizer.src_lang = source_language
        encoded = tokenizer(text, return_tensors="pt")

        with torch.inference_mode():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(target_language),
            )

        return tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]
    except (ValueError, RuntimeError, KeyError) as error:
        return f"Translation error: {error}"


interface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(
            label="Input text",
            placeholder="Enter text to translate...",
            lines=4,
        ),
        gr.Dropdown(
            choices=LANGUAGES,
            value="en",
            label="Source language",
            info="Choose the language of the input text.",
        ),
        gr.Dropdown(
            choices=LANGUAGES,
            value="es",
            label="Target language",
            info="Choose the language for the translation.",
        ),
    ],
    outputs=gr.Textbox(label="Translation", lines=4),
    title="Multilingual Translator",
    description=(
        "Translate text between 15 selected languages with Facebook's "
        "M2M100 model. Translation may be slower on CPU."
    ),
    examples=[
        ["Hello, how are you?", "en", "es"],
        ["Bonjour, comment ça va?", "fr", "zh"],
    ],
)


if __name__ == "__main__":
    interface.launch()
