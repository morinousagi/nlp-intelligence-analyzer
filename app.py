import gradio as gr

from src.pipeline import SigmaPipeline

# Initialize pipeline once during startup
pipeline = SigmaPipeline()


def analyze_text(text):
    """
    Run full SIGMA analysis pipeline and format results.
    """

    result = pipeline.analyze(text)

    formatted_output = []

    # Explicit Facts
    formatted_output.append("EXPLICIT FACTS")
    formatted_output.append("----------------")

    for fact in result["explicit_facts"]:
        formatted_output.append(
            f"• {fact['subject']} → {fact['action']} → {fact['object']}"
        )

    formatted_output.append("")

    # Implicit Inferences
    formatted_output.append("IMPLICIT INFERENCES")
    formatted_output.append("---------------------")

    for inference in result["implicit_inferences"]:
        formatted_output.append(
            f"• [{inference['label']}] "
            f"{inference['hypothesis']} "
            f"(Confidence: {inference['confidence']} | "
            f"Tier: {inference['confidence_tier']})"
        )

    formatted_output.append("")

    # Summary
    return "\n".join(formatted_output)


demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        lines=12,
        label="Intelligence Report Input",
        placeholder="Paste intelligence-style narrative text here..."
    ),
    outputs=gr.Textbox(
        lines=25,
        label="SIGMA Analysis Output"
    ),
    title="SIGMA — Structured Intelligence & Grounded Meaning Analyzer",
    description=(
        "SIGMA is an NLP-based intelligence reasoning prototype that extracts "
        "explicit facts, evaluates implicit inferences using Natural Language "
        "Inference (MNLI), and generates concise analytic summaries using "
        "pretrained transformer models."
    ),
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()