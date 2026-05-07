"""
fact_extractor.py

Module: Explicit Fact Extraction for SIGMA Intelligence Analyzer
Purpose:
    Extract structured, syntactic facts from narrative text.
    Uses spaCy transformer pipeline for:
        - Named Entity Recognition (NER)
        - Dependency parsing
        - Sentence segmentation
Output:
    - Entities (with labels)
    - Structured factual triples:
        (subject → action → object)

Design Philosophy:
    Conservative extraction.
    No speculative inference.
    Only syntactically grounded relations.
"""

import spacy
from spacy.cli import download

class FactExtractor:
    def __init__(self):
        """
        Load spaCy transformer-based English model.
        This model includes:
            - NER
            - POS tagging
            - Dependency parsing
        """
        #self.nlp = spacy.load("en_core_web_trf")  # transformer-based & very heavy
        #self.nlp = spacy.load("en_core_web_sm")   # lightweight version

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        

    def extract_entities(self, text):
        """
        Extract named entities from text.
        Returns:
            List[Dict] with:
                {
                    "text": entity string,
                    "label": entity type (ORG, PERSON, GPE, DATE, etc.)
                }
        """
        doc = self.nlp(text)

        return [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]

    def extract_structured_facts(self, text):
        """
        Extract structured factual triples from text.
        Strategy:
            - Iterate through sentences
            - Identify ROOT verb of each sentence
            - Extract:
                - Subject (nsubj / nsubjpass)
                - Object/complement structures
            - Expand subject and object to full phrase spans
        Returns: List[Dict]
        """

        doc = self.nlp(text)

        facts = []

        for sent in doc.sents:

            for token in sent:

                # Identify main predicate (ROOT verb)
                # note: some sentences whose ROOT is AUX or NOUN may still not extract perfectly
                if token.pos_ == "VERB" and token.dep_ == "ROOT":

                    subject = None
                    obj = None

                    # -----------------------------
                    # SUBJECT EXTRACTION
                    # -----------------------------
                    for child in token.children:

                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subject = self._expand_phrase(child)

                    # -----------------------------
                    # OBJECT / COMPLEMENT EXTRACTION
                    # -----------------------------

                    for child in token.children:

                        # Direct object / attribute / object predicate
                        if child.dep_ in ("dobj", "attr", "oprd"):

                            obj = self._expand_phrase(child)

                        # Clausal complements
                        elif child.dep_ in ("ccomp", "xcomp"):

                            obj = self._expand_phrase(child)

                            # Remove leading complementizer
                            if obj.startswith("that "):
                                obj = obj[5:]

                    # -----------------------------------
                    # Attach ROOT-level prep phrases
                    # -----------------------------------

                    if obj:

                        prep_phrases = []

                        for child in token.children:

                            if child.dep_ == "prep":

                                prep_text = self._expand_phrase(child)

                                # Avoid duplicate attachment
                                if prep_text not in obj:
                                    prep_phrases.append(prep_text)

                        if prep_phrases:
                            obj += " " + " ".join(prep_phrases)

                    # -----------------------------
                    # STORE FACT
                    # -----------------------------
                    facts.append({
                        "sentence": sent.text.strip(),
                        "subject": subject,
                        "action": token.lemma_,
                        "object": obj
                    })

        return facts
    
    def _expand_phrase(self, token):
        """
        Expand token to its full subtree span and normalize whitespace.
        Why:
            Dependency parsing identifies only head tokens.
            We want full noun phrases including modifiers.
        Example:
            Head token: "division"
            Expanded: "The cybersecurity division"
        Implementation:
            - Collect all tokens in subtree
            - Get span from first to last token
        """
        subtree = list(token.subtree)

        start = subtree[0].i
        end = subtree[-1].i + 1

        span_text = token.doc[start:end].text

        # Normalize whitespace (remove leading/trailing spaces and newlines)
        return " ".join(span_text.strip().split())

    def extract_sentences(self, text):
        """
        Return list of individual sentences.
        Useful for:
            - Candidate hypothesis generation
            - Summarization preprocessing
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]