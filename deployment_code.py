from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class TypoCorrector:
    def __init__(self, model_name, tokenizer_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Muat tokenizer dari folder tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
        
        # Muat model dengan penanganan ketidakcocokan ukuran parameter
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, ignore_mismatched_sizes=True)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model dan tokenizer dimuat dari {model_name} dan {tokenizer_dir}")
        
    def correct_text(self, text, max_length=128):
        # Encode input text dan lakukan koreksi
        input_ids = self.tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(input_ids)
        
        # Decode hasil koreksi menjadi teks
        corrected_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return corrected_text

# Ganti dengan path model .pt dan folder tokenizer Anda
MODEL_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'
TOKENIZER_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'

# Inisialisasi corrector dengan path model dan tokenizer yang benar
corrector = TypoCorrector(MODEL_PATH, TOKENIZER_PATH)

def correct_typo(text):
    original = text
    corrected = corrector.correct_text(text)
    return {"Original": original, "Corrected": corrected}
