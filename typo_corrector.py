from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

# Load model and tokenizer as global variables
MODEL_DIR = "MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector"
TOKENIZER_DIR = "MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector"
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_DIR)

class TypoCorrector:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def correct_text(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

class SentenceProcessor:
    def __init__(self):
        self.corrector = TypoCorrector()
    
    def process_sentences(self, text):
        sentences = text.split('.')
        corrected_sentences = [self.corrector.correct_text(sentence.strip()) for sentence in sentences if sentence.strip()]
        return '. '.join(corrected_sentences)