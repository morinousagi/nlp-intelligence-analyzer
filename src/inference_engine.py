"""
inference_engine.py

Controlled Implicit Inference using MNLI.
Compatible with transformers >=5.x

roberta-large-mnli Label-Meaning mapping is usually:
    LABEL_0	CONTRADICTION
    LABEL_1	NEUTRAL
    LABEL_2	ENTAILMENT
"""

from transformers import pipeline


class InferenceEngine:
    def __init__(self, threshold=0.80):
        """
        threshold: minimum confidence score required
        """
        self.threshold = threshold

        self.classifier = pipeline(
            "text-classification",
            model="roberta-large-mnli"
        )

    def validate_hypotheses(self, premise, hypotheses):
        """
        Validate candidate hypotheses using MNLI.
        Only keep entailments above confidence threshold.
        """

        validated = []

        for hypothesis in hypotheses:

            result = self.classifier(
                f"{premise} </s></s> {hypothesis}"
            )[0]

            label = result["label"]
            score = result["score"]

            # Some versions return LABEL_2 etc.
            if label.upper() in ["ENTAILMENT", "LABEL_2"] and score >= self.threshold:
                validated.append({
                    "hypothesis": hypothesis,
                    "confidence": round(score, 3)
                })

        return validated