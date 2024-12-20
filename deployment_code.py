import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import gdown
import os

class TypoCorrector:
    def __init__(self, model_file_id, tokenizer_file_id, model_dir="model", tokenizer_dir="tokenizer"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Unduh model dan tokenizer jika belum ada
            self.download_from_google_drive(model_file_id, model_dir)
            self.download_from_google_drive(tokenizer_file_id, tokenizer_dir)
            
            # Load tokenizer dan model
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir, ignore_mismatched_sizes=True)

            # Set decoder_start_token_id ke bos_token_id atau pad_token_id
            self.model.config.decoder_start_token_id = (
                self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
            )

            self.model.to(self.device)
            self.model.eval()

            st.success("Model dan tokenizer berhasil dimuat.")
        except Exception as e:
            st.error(f"Error saat memuat model atau tokenizer: {e}")
            raise
    @st.cache_resource
    def download_from_google_drive(self, file_id, destination):
        """
        Mengunduh file dari Google Drive
        """
        if not os.path.exists(destination):
            os.makedirs(destination)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)

    def correct_text(self, text, max_length=128):
        try:
            # Menambahkan prefix sebelum input
            prefixed_text = f"koreksi: {text}"

            # Encode input text dan lakukan koreksi
            input_ids = self.tokenizer.encode(
                prefixed_text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    decoder_start_token_id=self.model.config.decoder_start_token_id,
                    max_length=max_length,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

            # Decode hasil koreksi menjadi teks
            corrected_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return corrected_text
        except Exception as e:
            st.error(f"Error saat melakukan koreksi teks: {e}")
            return text  # Mengembalikan teks asli jika terjadi kesalahan

# Streamlit page configuration
st.set_page_config(
    page_title="Indonesian Typo Corrector",
    page_icon="✍️",
    layout="wide"
)

@st.cache_resource
def initialize_corrector():
    """
    Initialize the TypoCorrector dengan caching
    """
    # ID file model dan tokenizer dari Google Drive
    MODEL_FILE_ID = '1xJC2Dff8lL1BOWfF4hxSyTkefg7cnw9q'  # Ganti dengan ID file model Anda
    TOKENIZER_FILE_ID = ['1GSQSzrCwsg5_yOFAzB9GaFEFjCQtRN46', '1jVmo15bMy4GNGP4vRDyLk1b4H6bdZgcs', '1lXBqbmu05zkAseKRXSIX8-tW56ULczSp', '1vEgoBiMp3sovWYsPKKbG_uLzHOJyVWa4']  # Ganti dengan ID file tokenizer Anda

    try:
        return TypoCorrector(MODEL_FILE_ID, TOKENIZER_FILE_ID)
    except Exception as e:
        st.error(f"Inisialisasi gagal: {e}")
        return None

def main():
    st.title("Indonesian Typo Corrector")
    st.write("Masukkan teks yang ingin dikoreksi ejaannya.")

    # Inisialisasi corrector
    corrector = initialize_corrector()

    if corrector is None:
        st.error("Gagal memuat model. Silakan refresh halaman atau periksa koneksi internet Anda.")
        return

    # Input teks
    input_text = st.text_area("Teks Input:", height=150)

    if st.button("Koreksi Teks"):
        if not input_text.strip():
            st.warning("Mohon masukkan teks yang valid.")
            return

        with st.spinner("Sedang mengoreksi teks..."):
            try:
                result = corrector.correct_text(input_text)

                # Tampilkan hasil koreksi
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Teks Asli")
                    st.write(input_text)

                with col2:
                    st.subheader("Hasil Koreksi")
                    st.write(result)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengoreksi teks: {e}")

if __name__ == "__main__":
    main()
