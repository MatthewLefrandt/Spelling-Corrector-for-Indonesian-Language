import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Indonesian Typo Corrector",
    page_icon="✍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """
    Memuat model dan tokenizer dengan caching Streamlit
    """
    MODEL_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'
    TOKENIZER_PATH = 'MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with st.spinner('Loading model and tokenizer...'):
        try:
            tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)
            
            # Tambahkan decoder_start_token_id jika tidak ada
            model.config.decoder_start_token_id = model.config.decoder_start_token_id or model.config.bos_token_id or tokenizer.cls_token_id
            
            # Pastikan tokenizer memiliki cls_token_id
            if tokenizer.cls_token_id is None:
                tokenizer.add_special_tokens({'cls_token': '[CLS]'})
                tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
                model.resize_token_embeddings(len(tokenizer))
            
            model.to(device)
            model.eval()
            return model, tokenizer, device
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, None

def correct_text(text, model, tokenizer, device, max_length=128):
    """
    Melakukan koreksi teks menggunakan model
    """
    try:
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True).to(device)
        
        with torch.no_grad():
            output = model.generate(input_ids, decoder_start_token_id=model.config.decoder_start_token_id)
        
        corrected_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return corrected_text
    except Exception as e:
        st.error(f"Error during correction: {str(e)}")
        return text

def main():
    st.title("Indonesian Typo Corrector")
    st.write("Masukkan teks yang ingin dikoreksi ejaannya.")

    # Load model
    model, tokenizer, device = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please check your internet connection and try again.")
        return

    # Input text
    input_text = st.text_area("Teks Input:", height=150)
    
    if st.button("Koreksi Teks"):
        if not input_text:
            st.warning("Mohon masukkan teks terlebih dahulu.")
            return
            
        with st.spinner("Sedang mengoreksi teks..."):
            corrected = correct_text(input_text, model, tokenizer, device)
            
            # Tampilkan hasil
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Teks Asli")
                st.write(input_text)
            
            with col2:
                st.subheader("Hasil Koreksi")
                st.write(corrected)

if __name__ == "__main__":
    main()
