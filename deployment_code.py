import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class TypoCorrector:
    def __init__(self, model_name, tokenizer_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
            # Load model with size mismatch handling
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, ignore_mismatched_sizes=True)
            # Set decoder_start_token_id to bos_token_id or pad_token_id
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
        except Exception as e:
            st.error(f"Error loading model or tokenizer: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        st.success(f"Model and tokenizer loaded successfully")
        
    def correct_text(self, text, max_length=128):
        try:
            # Tambahkan prefix sebelum input
            prefixed_text = f"koreksi: {text}"
            
            # Encode input text and perform correction
            input_ids = self.tokenizer.encode(prefixed_text, return_tensors='pt', max_length=max_length, truncation=True).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    decoder_start_token_id=self.model.config.decoder_start_token_id,
                    max_length=max_length,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Decode correction result into text
            corrected_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return corrected_text
        except Exception as e:
            st.error(f"Error during text correction: {e}")
            return text  # Return original text if error occurs

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Indonesian Typo Corrector",
    page_icon="✍️",
    layout="wide"
)

@st.cache_resource
def initialize_corrector():
    """
    Initialize the TypoCorrector with caching
    """
    MODEL_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'
    TOKENIZER_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'
    
    try:
        return TypoCorrector(MODEL_PATH, TOKENIZER_PATH)
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return None

def main():
    st.title("Indonesian Typo Corrector")
    st.write("Masukkan teks yang ingin dikoreksi ejaannya.")

    # Initialize corrector
    corrector = initialize_corrector()
    
    if corrector is None:
        st.error("Failed to initialize the model. Please refresh the page or check your internet connection.")
        return

    # Input text
    input_text = st.text_area("Teks Input:", height=150)
    
    if st.button("Koreksi Teks"):
        if not input_text:
            st.warning("Mohon masukkan teks terlebih dahulu.")
            return
            
        with st.spinner("Sedang mengoreksi teks..."):
            try:
                result = corrector.correct_text(input_text)
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Teks Asli")
                    st.write(input_text)
                
                with col2:
                    st.subheader("Hasil Koreksi")
                    st.write(result)
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengoreksi teks: {str(e)}")

if __name__ == "__main__":
    main()
