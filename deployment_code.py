import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set page configuration
st.set_page_config(
    page_title="Indonesian Spell Checker",
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
    """Correct spelling in the input text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def main():
    # Header
    st.title("🔤 Pengoreksi Ejaan Bahasa Indonesia")
    st.write("Masukkan teks dengan kesalahan ejaan, dan aplikasi akan memperbaikinya.")
    
    # Load model and tokenizer
    with st.spinner("Memuat model... Mohon tunggu sebentar..."):
        model, tokenizer = load_model_and_tokenizer()
    
    # Text input
    input_text = st.text_area(
        "Masukkan teks:",
        height=150,
        placeholder="Contoh: sya skt kpala krna blm mkn..."
    )
    
    # Correction button
    if st.button("Perbaiki Ejaan"):
        if input_text.strip():
            with st.spinner("Sedang memperbaiki ejaan..."):
                try:
                    corrected_text = correct_spelling(input_text, model, tokenizer)
                    
                    # Display results
                    st.subheader("Hasil Perbaikan:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text_area("Teks Asli:", value=input_text, height=150, disabled=True)
                    
                    with col2:
                        st.text_area("Teks yang Diperbaiki:", value=corrected_text, height=150, disabled=True)
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
        else:
            st.warning("Mohon masukkan teks terlebih dahulu!")

if __name__ == "__main__":
    main()
