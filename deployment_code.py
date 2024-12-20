from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class TypoCorrector:
    def __init__(self, model_name, tokenizer_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load tokenizer and model, and handle errors if they occur
        try:
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
            # Load model with size mismatch handling
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, ignore_mismatched_sizes=True)
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model and tokenizer loaded from {model_name} and {tokenizer_dir}")
        
    def correct_text(self, text, max_length=128):
        try:
            # Encode input text and perform correction
            input_ids = self.tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(input_ids)
            
            # Decode correction result into text
            corrected_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return corrected_text
        except Exception as e:
            print(f"Error during text correction: {e}")
            return text  # Return the original text if an error occurs

# Update with your correct model and tokenizer path
MODEL_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'
TOKENIZER_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'

# Initialize corrector
try:
    corrector = TypoCorrector(MODEL_PATH, TOKENIZER_PATH)
except Exception as e:
    print(f"Initialization failed: {e}")

def correct_typo(text):
    try:
        original = text
        corrected = corrector.correct_text(text)
        return {"Original": original, "Corrected": corrected}
    except Exception as e:
        print(f"Error correcting text: {e}")
        return {"Original": text, "Corrected": text}
