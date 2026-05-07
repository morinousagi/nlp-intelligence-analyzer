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

## Live Demo

Try the interactive SIGMA reasoning tool here:
[Hugging Face Space Link](https://huggingface.co/spaces/morinousagi/nlp-intelligence-analyzer)

## Model Stack

- `roberta-large-mnli` (entailment detection)

- `facebook/bart-large-cnn` (summarization)

- `spaCy en_core_web_sm` (entity extraction) - extraction uses syntactic dependency parsing to avoid speculative relation generation and maintain analytic defensibility.

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
#### Architecture of Inference Layer
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

## Files
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