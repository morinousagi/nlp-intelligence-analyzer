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

    formatted_output.append("## EXPLICIT FACTS\n")

    for fact in result["explicit_facts"]:
        formatted_output.append(
            f"- **{fact['subject']}** → `{fact['action']}` → {fact['object']}"
        )

    formatted_output.append("\n## IMPLICIT INFERENCES\n")

    for inference in result["implicit_inferences"]:
        formatted_output.append(
            f"- **[{inference['label']}]** "
            f"{inference['hypothesis']}  \n"
            f"  Confidence: `{inference['confidence']}` "
            f"({inference['confidence_tier']})"
        )

    formatted_output.append("\n## ANALYTIC SUMMARY\n")

    formatted_output.append(result["summary"])

    return "\n".join(formatted_output)


with gr.Blocks() as demo:

    gr.Markdown("# SIGMA: Structured Intelligence & Grounded Meaning Analyzer")
    gr.Markdown(
        "Extract explicit facts, evaluate implicit inferences, "
        "and generate intelligence-style summaries."
    )

    input_text = gr.Textbox(
        lines=12,
        label="Intelligence Report Input",
        placeholder="Paste intelligence-style narrative text here..."
    )

    submit_btn = gr.Button("Analyze")

    output_text = gr.Markdown()

    submit_btn.click(
        fn=analyze_text,
        inputs=input_text,
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()