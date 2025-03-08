import streamlit as st
import torch
import io
import numpy as np
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from config import MODEL_ID, DEFAULT_STEPS, MAX_STEPS
from utils import validate_token
from model import ImageGenerator

# ------------------ Initialize Session State ------------------ #
def init_session_state():
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = None
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "device" not in st.session_state:
        if torch.cuda.is_available():
            st.session_state.device = "cuda"
        elif torch.backends.mps.is_available():
            st.session_state.device = "mps"
        else:
            st.session_state.device = "cpu"

    # Load models if not already loaded
    if "processor" not in st.session_state:
        try:
            st.session_state.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        except Exception as e:
            st.error(f"Failed to load Wav2Vec2 processor: {str(e)}")
            return
    if "model" not in st.session_state:
        try:
            st.session_state.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(st.session_state.device)
        except Exception as e:
            st.error(f"Failed to load Wav2Vec2 model: {str(e)}")
            return

# ------------------ Transcribe Audio ------------------ #
def transcribe_audio(audio_bytes):
    """Convert speech to text using Wav2Vec2."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize audio samples to [-1, 1]
        samples = samples / np.iinfo(np.int16).max  

        # Convert to tensor and ensure correct shape
        waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len)

        # Move to correct device
        input_values = st.session_state.processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(st.session_state.device)

        # Ensure proper shape for model inference
        input_values = input_values.squeeze(1)  # Fix tensor shape issue

        with torch.no_grad():
            logits = st.session_state.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = st.session_state.processor.batch_decode(predicted_ids)[0]

        return transcription
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")
        return None

# ------------------ Streamlit UI Setup ------------------ #
def setup_page():
    st.set_page_config(page_title="AI Audio-to-Image", layout="wide")
    st.title("🎨 AI Audio-to-Image Generator")
    st.write("Upload an audio file, transcribe it, and generate an image based on the text.")
    st.sidebar.markdown(f"**Device:** `{st.session_state.device}`")

# ------------------ Handle Hugging Face Token ------------------ #
def token_input():
    with st.form("token_form"):
        token = st.text_input("Enter your Hugging Face token:", type="password")
        token_submit = st.form_submit_button("Submit Token")
        
        if token_submit:
            if validate_token(token):
                st.session_state.hf_token = token
                st.session_state.generator = ImageGenerator(st.session_state.hf_token, MODEL_ID)
                st.success("Token validated successfully!")
                st.rerun()
            else:
                st.error("Invalid token. Please check your token and try again.")

        st.info("Get your Hugging Face token at: [Hugging Face Tokens](https://huggingface.co/settings/tokens)")
    st.stop()

# ------------------ Main Application ------------------ #
def main():
    init_session_state()
    setup_page()

    if st.session_state.hf_token is None:
        token_input()

    if st.sidebar.button("Logout (Clear Token)"):
        st.session_state.hf_token = None
        st.session_state.generator = None
        st.rerun()

    # Upload and Transcribe Audio
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with st.spinner("Transcribing audio..."):
            audio_bytes = uploaded_file.read()
            transcription = transcribe_audio(audio_bytes)
            if transcription is None:
                return
            st.success("Transcription complete!")
            st.text_area("Transcribed Text", transcription)

        # Generate Image
        if st.button("Generate Image"):
            if st.session_state.generator is None:
                if st.session_state.hf_token is None:
                    st.error("Please enter a valid Hugging Face token first!")
                    return
                else:
                    st.session_state.generator = ImageGenerator(st.session_state.hf_token, MODEL_ID)
            
            with st.spinner("Generating image..."):
                try:
                    generated_image = st.session_state.generator.generate(transcription, DEFAULT_STEPS)
                    st.image(generated_image, caption=f"Generated Image for: {transcription}", use_column_width=True)

                    buf = io.BytesIO()
                    generated_image.save(buf, format="PNG")
                    st.download_button("Download Image", buf.getvalue(), "generated_image.png", "image/png")
                except Exception as e:
                    st.error(f"An error occurred while generating the image: {str(e)}")

# ------------------ Run Streamlit App ------------------ #
if __name__ == "__main__":
    main()

