# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch

# Set page configuration
st.set_page_config(
    page_title="Koreksi Ejaan Bahasa Indonesia",
    page_icon="✍️",
    layout="centered"
)

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer from HuggingFace"""
    tokenizer = AutoTokenizer.from_pretrained('MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector')
    model = AutoModelForSeq2SeqGeneration.from_pretrained('MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector')
    return model, tokenizer

def correct_spelling(text, model, tokenizer):
    """Correct spelling using the loaded model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def main():
    st.title("Koreksi Ejaan Bahasa Indonesia 📝")
    st.write("Aplikasi ini akan membantu Anda memperbaiki kesalahan pengetikan dalam bahasa Indonesia.")
    
    # Load model and tokenizer
    try:
        with st.spinner("Memuat model... Mohon tunggu sebentar."):
            model, tokenizer = load_model_and_tokenizer()
        st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        return
    
    # Text input
    text_input = st.text_area(
        "Masukkan teks yang ingin dikoreksi:",
        height=150,
        placeholder="Contoh: sya ska mkn nasi greng"
    )
    
    # Add correction button
    if st.button("Koreksi Teks", type="primary"):
        if text_input.strip():
            try:
                with st.spinner("Sedang mengoreksi teks..."):
                    corrected_text = correct_spelling(text_input, model, tokenizer)
                
                # Display results
                st.subheader("Hasil Koreksi:")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Teks Asli:", value=text_input, height=100, disabled=True)
                with col2:
                    st.text_area("Teks Terkoreksi:", value=corrected_text, height=100, disabled=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengoreksi teks: {str(e)}")
        else:
            st.warning("Mohon masukkan teks terlebih dahulu!")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Dibuat dengan ❤️ menggunakan Streamlit dan Model T5 untuk Koreksi Ejaan Bahasa Indonesia</p>
            <p>Model by: MatthewLefrandt</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
