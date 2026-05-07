"""
pipeline.py

Core orchestration layer for SIGMA Intelligence Analyzer.

Combines:
    - Explicit fact extraction
    - Hypothesis generation
    - MNLI validation
    - Summarization
"""

from .fact_extractor import FactExtractor
from .inference_engine import InferenceEngine
from .summarizer import BriefSummarizer


class SigmaPipeline:

    def __init__(self):
        self.fact_extractor = FactExtractor()
        self.inference_engine = InferenceEngine()
        self.summarizer = BriefSummarizer()

    def analyze(self, text):
        """
        Full intelligence analysis workflow.
        """

        structured_facts = self.fact_extractor.extract_structured_facts(text)

        hypotheses = self._generate_hypotheses(structured_facts)

        validated_inferences = self.inference_engine.validate_hypotheses(
            text,
            hypotheses
        )

        summary = self.summarizer.summarize(text)

        return {
            "explicit_facts": structured_facts,
            "implicit_inferences": validated_inferences,
            "summary": summary
        }

    def _generate_hypotheses(self, structured_facts):
        """
        Internal method to generate normalized hypotheses
        from extracted structured facts.
        Purpose:
            Convert extracted factual structures into
            standalone propositions suitable for MNLI validation.
        """

        hypotheses = []

        reporting_verbs = {
            "report",
            "confirm",
            "state",
            "announce",
            "detect",
            "record",
            "identify"
        }

        assessment_verbs = {
            "believe",
            "assess",
            "suspect",
            "estimate"
        }

        for fact in structured_facts:

            action = fact["action"]
            obj = fact["object"]

            if not obj:
                continue

            obj = obj.strip()

            # -----------------------------------
            # NEGATED ACTIONS
            # -----------------------------------

            if action.startswith("not "):

                clean_action = action.replace("not ", "")

                if clean_action == "attribute":
                    hypotheses.append(f"It is not confirmed that {obj}.")

                else:
                    hypotheses.append(f"It did not occur that {obj}.")

            # -----------------------------------
            # REPORTING / OBSERVATION VERBS
            # -----------------------------------

            elif action in reporting_verbs:

                hypotheses.append(f"There was {obj}.")

            # -----------------------------------
            # ANALYTIC / ASSESSMENT VERBS
            # -----------------------------------

            elif action in assessment_verbs:

                hypotheses.append(obj)

            # -----------------------------------
            # DEFAULT FALLBACK
            # -----------------------------------

            else:

                hypotheses.append(obj)

        return hypotheses