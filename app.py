import gradio as gr
from pipeline import SigmaPipeline

# Initialize pipeline once (important)
pipeline = SigmaPipeline()

def analyze_text(text):
    result = pipeline.analyze(text)
    return result

demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=10, placeholder="Enter intelligence-style passage..."),
    outputs="json",
    title="Sigma: Intelligence Reasoning Prototype",
    description="Extracts explicit facts, validates implicit inferences, and generates summary using pretrained NLP models."
)

if __name__ == "__main__":
    demo.launch()