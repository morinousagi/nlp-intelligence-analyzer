from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class BriefSummarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-large-cnn"
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn"
        ).to(self.device)

    def summarize(self, text, max_length=130, min_length=40):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True
        )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary