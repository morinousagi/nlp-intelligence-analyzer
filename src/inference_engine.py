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
        Returns:
            List of inference classifications.
        """

        validated = []

        label_mapping = {
            "LABEL_0": "CONTRADICTION",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "ENTAILMENT"
        }

        for hypothesis in hypotheses:

            result = self.classifier(
                f"{premise} </s></s> {hypothesis}"
            )[0]

            raw_label = result["label"]
            score = result["score"]

            label = label_mapping.get(raw_label, raw_label)

            validated.append({
                "hypothesis": hypothesis,
                "label": label,
                "confidence": round(score, 3),
                "confidence_tier": self._confidence_tier(score)
            })

        return validated
    
    def _confidence_tier(self, score):
        """
        Convert numeric confidence into analyst-friendly tier.
        """

        if score >= 0.95:
            return "High"

        elif score >= 0.80:
            return "Moderate"

        return "Low"