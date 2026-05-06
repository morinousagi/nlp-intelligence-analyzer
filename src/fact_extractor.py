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
            nlp = spacy.load("en_core_web_sm")
        except:
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        

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
                - Object (dobj / attr / prep / ccomp)
            - Expand subject and object to full phrase spans
        Returns:
            List[Dict] with:
                {
                    "sentence": original sentence,
                    "subject": full subject phrase,
                    "action": lemmatized root verb,
                    "object": full object phrase
                }
        """

        doc = self.nlp(text)
        facts = []

        for sent in doc.sents:
            for token in sent:

                # Identify main predicate (ROOT verb)
                if token.pos_ == "VERB" and token.dep_ == "ROOT":

                    subject = None
                    obj = None

                    # Search for grammatical subject
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subject = self._expand_phrase(child)

                    # Search for object or complement
                    for child in token.children:
                        if child.dep_ in ("dobj", "attr", "prep", "ccomp"):
                            obj = self._expand_phrase(child)

                    facts.append({
                        "sentence": sent.text.strip(),
                        "subject": subject,
                        "action": token.lemma_,  # Use lemma for normalized verb
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