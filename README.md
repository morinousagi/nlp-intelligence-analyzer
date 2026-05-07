---
title: Structured Intelligence & Grounded Meaning Analyzer
emoji: 👀
colorFrom: yellow
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Structured Intelligence & Grounded Meaning Analyzer

This project was developed using prompt engineering techniques with ChatGPT GPT-5.5.

The application is deployed using Hugging Face Spaces.

Try the interactive SIGMA reasoning tool here:
[Hugging Face Space Link](https://huggingface.co/spaces/morinousagi/nlp-intelligence-analyzer)

---

## Model Stack

### Fact Extraction 
- `spaCy en_core_web_sm`
- Explicit fact extraction, extracts structured factual triples:
`Subject → Action → Object`


### Inference Validation 
- `roberta-large-mnli` (Multi-Genre Natural Language Inference)
- Transformer model to evaluate candidate hypotheses against the original passage, classifying each hypothesis as:
   - ENTAILMENT
   - NEUTRAL
   - CONTRADICTION

- Confidence scores are derived from the model's softmax probability distribution over the three inference classes.

### Summarization
- `facebook/bart-large-cnn`
- Generates concise analytic summaries


### Technical Notes
- Uses conservative syntactic extraction to avoid speculative relation generation
- Separates confirmed facts from analytical assessments
- Built entirely with pretrained transformer models
- Optimized for CPU deployment on Hugging Face Spaces

---

## Architecture
```
Input Text
   ↓
[NER + Relation Extraction]
   ↓
Explicit Fact List
   ↓
[NLI Model - Entailment Testing]
   ↓
Validated Implicit Inferences
   ↓
[Summarization Model]
   ↓
Intelligence Brief Output
```
### Architecture of Inference Layer
```
Structured Facts
       ↓
Embedded Clause Extraction
       ↓
Standalone Hypothesis Generation
       ↓
MNLI Entailment Test
       ↓
Filter (confidence threshold)
       ↓
Implicit Inference Output
```

---

## Project Structure
```
/
├── src/                     # Core NLP logic
│   ├── __init__.py
│   ├── fact_extractor.py
│   ├── inference_engine.py
│   ├── summarizer.py
│   ├── pipeline.py
│
├── app.py                   # UI / deployment entrypoint
├── requirements.txt
└── README.md
```
