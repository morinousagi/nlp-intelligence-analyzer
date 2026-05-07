import gradio as gr

from src.pipeline import SigmaPipeline

# Initialize pipeline once during startup
pipeline = SigmaPipeline()


def analyze_text(text):
    """
    Run full SIGMA intelligence analysis pipeline.
    """
    result = pipeline.analyze(text)
    return result


demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Enter intelligence-style passage..."
    ),
    outputs="json",
    title="SIGMA: Intelligence Reasoning Prototype",
    description=(
        "Extracts explicit facts, validates implicit inferences, "
        "and generates concise intelligence-style summaries "
        "using pretrained NLP models."
    )
)

if __name__ == "__main__":
    demo.launch()