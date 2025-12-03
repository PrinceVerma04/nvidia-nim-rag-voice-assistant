import streamlit as st
import os
from streamlit_mic_recorder import mic_recorder
from pathlib import Path
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import time
import wave
import tempfile
import numpy as np

# Import Riva client for ASR
try:
    import riva.client
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False
    st.warning("Riva client not installed. Run: pip install nvidia-riva-client")

# Set page config
st.set_page_config(page_title="Nvidia NIM RAG Demo", page_icon="", layout="wide")

# Function to load API keys from file
def load_api_keys():
    api_key_file = Path(__file__).parent / "API_Key.txt"
    
    if not api_key_file.exists():
        st.error(f"API_Key.txt not found at: {api_key_file}")
        st.stop()
    
    try:
        with open(api_key_file, 'r') as f:
            lines = f.readlines()
            
        api_keys = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if '=' in line:
                key_name, key_value = line.split('=', 1)
                key_name = key_name.strip()
                key_value = key_value.strip().strip('"').strip("'")
                
                if key_value and key_value.startswith('nvapi-'):
                    api_keys[key_name] = key_value
        
        # Check if we have both keys
        if 'LLM_API_KEY' not in api_keys:
            st.error("LLM_API_KEY not found in API_Key.txt")
            st.stop()
        
        if 'CANARY_API_KEY' not in api_keys:
            st.warning("CANARY_API_KEY not found in API_Key.txt. Voice features will be disabled.")
            api_keys['CANARY_API_KEY'] = None
        
        return api_keys
        
    except Exception as e:
        st.error(f"Error reading API keys: {str(e)}")
        st.stop()

# Load API keys
api_keys = load_api_keys()
nvidia_api_key = api_keys['LLM_API_KEY']
canary_api_key = api_keys.get('CANARY_API_KEY')

# Show masked API keys in sidebar for verification
st.sidebar.success(f"LLM API Key: {nvidia_api_key[:10]}...{nvidia_api_key[-4:]}")
if canary_api_key:
    st.sidebar.success(f"Canary API Key: {canary_api_key[:10]}...{canary_api_key[-4:]}")
else:
    st.sidebar.warning("Canary API Key not configured")

# Initialize LLM
llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-v3.1-terminus",
    temperature=0.5,
    top_p=0.7,
    max_completion_tokens=8192,
    api_key=nvidia_api_key
)

def vector_embedding():
    """Create vector embeddings from PDF documents"""
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector embeddings..."):
            # Pass API key to embeddings
            st.session_state.embeddings = NVIDIAEmbeddings(
                model="nvidia/nv-embedqa-e5-v5",
                api_key=nvidia_api_key
            )
            
            pdf_path = Path(__file__).parent / "pdf"
            st.session_state.loader = PyPDFDirectoryLoader(str(pdf_path))
            st.session_state.documents = st.session_state.loader.load()
            
            if not st.session_state.documents:
                st.error("No PDF documents found in the pdf folder!")
                return False
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.documents[:30]
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, 
                st.session_state.embeddings
            )
            return True
    return True

def format_docs(docs):
    """Format documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_response(question):
    """Get response from RAG chain - UNCHANGED from original"""
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the provided context only.
        
Context:
{context}

Question: {question}

Answer:"""
    )
    
    # Build RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get answer and source docs
    answer = rag_chain.invoke(question)
    source_docs = retriever.invoke(question)
    
    return answer, source_docs

import io
import wave

def convert_to_wav(audio_bytes, sample_rate=16000, channels=1, sample_width=2):
    """
    Convert raw audio bytes to WAV format with proper headers
    Only use this for raw PCM data from mic_recorder
    """
    try:
        # Check if already a WAV file (starts with 'RIFF')
        if audio_bytes[:4] == b'RIFF':
            # Already in WAV format, return as-is
            return audio_bytes
        
        # Create WAV file in memory for raw PCM data
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        
        wav_buffer.seek(0)
        return wav_buffer.read()
        
    except Exception as e:
        st.error(f"Error converting to WAV: {str(e)}")
        return None
def read_wav_info(audio_bytes):
    """
    Read WAV file format information and convert to mono if needed
    Returns: (sample_rate, channels, sample_width, audio_data)
    """
    try:
        wav_buffer = io.BytesIO(audio_bytes)
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            audio_data = wav_file.readframes(wav_file.getnframes())
        
        # Convert stereo to mono if needed
        if channels == 2:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Reshape to separate left and right channels
            audio_array = audio_array.reshape(-1, 2)
            
            # Average the two channels to create mono
            mono_audio = audio_array.mean(axis=1).astype(np.int16)
            
            # Convert back to bytes
            audio_data = mono_audio.tobytes()
            channels = 1  # Now mono
        
        return sample_rate, channels, sample_width, audio_data
    except Exception as e:
        st.error(f"Error reading WAV info: {str(e)}")
        return None, None, None, None



def transcribe_audio_canary(audio_bytes, source_language="en-US", is_uploaded=False):
    """
    Transcribe audio using NVIDIA Canary ASR model via gRPC
    
    Args:
        audio_bytes: Audio data in bytes
        source_language: Language code (default: en-US)
        is_uploaded: True if from file upload, False if from mic recorder
    
    Returns:
        Transcribed text string
    """
    if not RIVA_AVAILABLE:
        return "Error: Riva client not installed. Please run: pip install nvidia-riva-client"
    
    if not canary_api_key:
        return "Error: Canary API key not configured in API_Key.txt"
    
    try:
        # Handle uploaded files vs recorded audio differently
        if is_uploaded:
            # Uploaded file - read WAV info (converts stereo to mono)
            sample_rate, channels, sample_width, audio_data = read_wav_info(audio_bytes)
            
            if sample_rate is None:
                return "Error: Could not read audio file format"
            
            # Rebuild WAV file with mono audio
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)  # Now 1 (mono)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
        else:
            # Recorded audio from mic_recorder - check if it's already WAV
            if audio_bytes[:4] == b'RIFF':
                # Already WAV format - read and convert like uploaded files
                sample_rate, channels, sample_width, audio_data = read_wav_info(audio_bytes)
                
                if sample_rate is None:
                    return "Error: Could not read recorded audio format"
                
                # Check audio duration (must be > 0.5 seconds)
                duration = len(audio_data) / (sample_rate * sample_width * channels)
                if duration < 0.5:
                    return f"Error: Audio too short ({duration:.2f}s). Please record at least 1 second of speech."
                
                # Rebuild WAV file with mono audio
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(channels)  # Now 1 (mono)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
                wav_buffer.seek(0)
                wav_bytes = wav_buffer.read()
            else:
                # Raw PCM data - convert to WAV
                sample_rate = 16000
                channels = 1
                wav_bytes = convert_to_wav(audio_bytes, sample_rate=sample_rate, channels=channels)
                
                if not wav_bytes:
                    return "Error: Failed to convert audio to WAV format"
        
        # Riva server configuration
        auth = riva.client.Auth(
            uri="grpc.nvcf.nvidia.com:443",
            use_ssl=True,
            metadata_args=[
                ["function-id", "b0e8b4a5-217c-40b7-9b96-17d84e666317"],
                ["authorization", f"Bearer {canary_api_key}"]
            ]
        )
        
        # Create ASR service
        asr_service = riva.client.ASRService(auth)
        
        # Configure ASR - always use mono (1 channel)
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=int(sample_rate),
            language_code=source_language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            verbatim_transcripts=False,
            audio_channel_count=1  # Always 1 (mono)
        )
        
        # Perform transcription
        response = asr_service.offline_recognize(wav_bytes, config)
        
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            return transcript
        else:
            return "No speech detected in audio"
            
    except Exception as e:
        return f"Transcription error: {str(e)}"


def save_uploaded_audio_as_wav(uploaded_file):
    """
    Save uploaded audio file and ensure it's in WAV format
    
    Returns:
        bytes: Audio data in WAV format
    """
    try:
        # Read the uploaded file
        audio_bytes = uploaded_file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Read back as bytes
        with open(tmp_path, 'rb') as f:
            wav_bytes = f.read()
        
        # Clean up
        os.unlink(tmp_path)
        
        return wav_bytes
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
st.title("Nvidia NIM RAG Demo with Voice")
st.markdown("Ask questions using text or voice input!")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("Initialize Vector Store", use_container_width=True):
        success = vector_embedding()
        if success:
            st.success("Vector store ready!")
            st.session_state.vector_initialized = True
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Show stats
    if "vectors" in st.session_state:
        st.metric("Documents Loaded", len(st.session_state.documents))
        st.metric("📝 Text Chunks", len(st.session_state.final_documents))
    
    st.markdown("---")
    
    # Voice settings
    st.subheader("🎤 Voice Settings")
    language_options = {
        "English (US)": "en-US",
        "Spanish (ES)": "es-ES",
        "French (FR)": "fr-FR",
        "German (DE)": "de-DE",
        "Italian (IT)": "it-IT",
        "Portuguese (BR)": "pt-BR",
        "Japanese (JP)": "ja-JP",
        "Korean (KR)": "ko-KR",
        "Mandarin (CN)": "zh-CN",
        "Hindi (IN)": "hi-IN"
    }
    
    selected_language = st.selectbox(
        "Select Language",
        options=list(language_options.keys()),
        index=0
    )
    language_code = language_options[selected_language]
    
    st.markdown("---")
    st.caption("Tip: Initialize the vector store before asking questions")

# Create tabs for Text and Voice input
tab1, tab2 = st.tabs([" Text Chat", " Voice Chat"])

# ==================== TAB 1: TEXT CHAT ====================
with tab1:
    st.markdown("### Text-based Q&A")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show source documents for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(" View Source Documents"):
                    for i, doc in enumerate(message["sources"]):
                        st.write(f"**Document {i+1}:**")
                        st.write(doc.page_content)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Source: {doc.metadata}")
                        st.markdown("---")
# ==================== TAB 2: VOICE CHAT ====================
with tab2:
    st.markdown("### Voice-based Q&A")
    
    if not RIVA_AVAILABLE:
        st.error("Riva client is not installed. Please run: `pip install nvidia-riva-client`")
    elif not canary_api_key:
        st.error("Canary API key not found in API_Key.txt.")
        st.info("Get your Canary API key from: https://build.nvidia.com/nvidia/canary-1b-asr")
    else:
        st.info(f" Selected Language: **{selected_language}** ({language_code})")
        
        # Create two columns
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🎤 Record Audio")
            # Use mic_recorder instead of st.audio_input
            audio = mic_recorder(
                start_prompt=" Start Recording",
                stop_prompt=" Stop Recording",
                just_once=False,
                use_container_width=True,
                key='voice_recorder'
            )
            
            if audio:
                st.success(" Audio recorded!")
                st.audio(audio['bytes'])
                
                # if st.button("🎤 Transcribe & Ask", key="transcribe_recorded", use_container_width=True):
                #     if "vectors" not in st.session_state:
                #         st.warning("⚠️ Please initialize the Vector Store first!")
                #     else:
                #         with st.spinner("🎧 Transcribing audio..."):
                if st.button(" Transcribe & Ask", key="transcribe_recorded", use_container_width=True):
                    if "vectors" not in st.session_state:
                        st.warning(" Please initialize the Vector Store first!")
                    else:
                        # DEBUG: Show audio properties
                        st.info(f" Audio size: {len(audio['bytes'])} bytes")
                        st.info(f" First 4 bytes: {audio['bytes'][:4]}")
                        
                        # Check if it's WAV format
                        if audio['bytes'][:4] == b'RIFF':
                            try:
                                wav_buffer = io.BytesIO(audio['bytes'])
                                with wave.open(wav_buffer, 'rb') as wav_file:
                                    st.info(f" Sample rate: {wav_file.getframerate()} Hz")
                                    st.info(f" Channels: {wav_file.getnchannels()}")
                                    st.info(f" Sample width: {wav_file.getsampwidth()} bytes")
                                    st.info(f" Duration: {wav_file.getnframes() / wav_file.getframerate():.2f} seconds")
                            except Exception as e:
                                st.error(f"Error reading WAV: {e}")
                        
                        # Save to disk for inspection
                        debug_path = Path(__file__).parent / "debug_recorded.wav"
                        with open(debug_path, 'wb') as f:
                            f.write(audio['bytes'])
                        st.info(f" Saved to: {debug_path}")
                        
                        with st.spinner(" Transcribing audio..."):

                            # Use audio bytes directly
                            # transcribed_text = transcribe_audio_canary(audio['bytes'], language_code)
                            transcribed_text = transcribe_audio_canary(audio['bytes'], language_code, is_uploaded=False)

                            
                            st.success(" Transcription complete!")
                            st.markdown(f"**Transcribed Question:** _{transcribed_text}_")
                            
                            if not transcribed_text.startswith("Error:") and transcribed_text != "No speech detected in audio":
                                st.session_state.messages.append({
                                    "role": "user", 
                                    "content": f"🎤 [Voice]: {transcribed_text}"
                                })
                                
                                with st.spinner(" Generating answer..."):
                                    try:
                                        start_time = time.process_time()
                                        answer, source_docs = get_rag_response(transcribed_text)
                                        response_time = time.process_time() - start_time
                                        
                                        st.markdown("### Answer:")
                                        st.markdown(answer)
                                        st.caption(f"Response time: {response_time:.2f}s")
                                        
                                        with st.expander("View Source Documents"):
                                            for i, doc in enumerate(source_docs):
                                                st.write(f"**Document {i+1}:**")
                                                st.write(doc.page_content)
                                                if hasattr(doc, 'metadata') and doc.metadata:
                                                    st.caption(f"Source: {doc.metadata}")
                                                st.markdown("---")
                                        
                                        st.session_state.messages.append({
                                            "role": "assistant", 
                                            "content": answer,
                                            "sources": source_docs
                                        })
                                        
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                            else:
                                st.error(transcribed_text)
        
        with col_right:
            st.subheader("Upload Audio File")
            uploaded_audio = st.file_uploader(
                "Or upload an audio file",
                type=["wav", "mp3", "flac", "opus"],
                help="Upload a pre-recorded audio file",
                key="audio_uploader"
            )
            
            if uploaded_audio:
                st.audio(uploaded_audio)
                
                if st.button("Transcribe Uploaded", key="transcribe_uploaded", use_container_width=True):
                    if "vectors" not in st.session_state:
                        st.warning("Please initialize the Vector Store first!")
                    else:
                        with st.spinner("Transcribing audio..."):
                            audio_bytes = save_uploaded_audio_as_wav(uploaded_audio)
                            
                            if audio_bytes:
                                # transcribed_text = transcribe_audio_canary(audio_bytes, language_code)
                                transcribed_text = transcribe_audio_canary(audio_bytes, language_code, is_uploaded=True)

                                
                                st.success("Transcription complete!")
                                st.markdown(f"**Transcribed Question:** _{transcribed_text}_")
                                
                                if not transcribed_text.startswith("Error:") and transcribed_text != "No speech detected in audio":
                                    st.session_state.messages.append({
                                        "role": "user", 
                                        "content": f"[Voice]: {transcribed_text}"
                                    })
                                    
                                    with st.spinner(" Generating answer..."):
                                        try:
                                            start_time = time.process_time()
                                            answer, source_docs = get_rag_response(transcribed_text)
                                            response_time = time.process_time() - start_time
                                            
                                            st.markdown("### Answer:")
                                            st.markdown(answer)
                                            st.caption(f"Response time: {response_time:.2f}s")
                                            
                                            with st.expander(" View Source Documents"):
                                                for i, doc in enumerate(source_docs):
                                                    st.write(f"**Document {i+1}:**")
                                                    st.write(doc.page_content)
                                                    if hasattr(doc, 'metadata') and doc.metadata:
                                                        st.caption(f"Source: {doc.metadata}")
                                                    st.markdown("---")
                                            
                                            st.session_state.messages.append({
                                                "role": "assistant", 
                                                "content": answer,
                                                "sources": source_docs
                                            })
                                            
                                        except Exception as e:
                                            st.error(f" Error: {str(e)}")
                                else:
                                    st.error(transcribed_text)
        
        st.markdown("---")
        st.markdown("####  Recent Voice Conversations")
        
        voice_messages = [msg for msg in st.session_state.messages if "🎤" in msg.get("content", "")]
        
        if voice_messages:
            for i, msg in enumerate(voice_messages[-5:]):
                if msg["role"] == "user":
                    st.markdown(f"**Q{i+1}:** {msg['content'].replace('🎤 [Voice]: ', '')}")
        else:
            st.info("No voice queries yet. Record or upload audio to get started!")

# ==================== CHAT INPUT (OUTSIDE TABS) ====================
# Move chat input here - AFTER the tabs context
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Check if vector store is initialized
    if "vectors" not in st.session_state:
        st.warning(" Please initialize the Vector Store first using the button in the sidebar!")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start_time = time.process_time()
                    answer, source_docs = get_rag_response(prompt)
                    response_time = time.process_time() - start_time
                    
                    # Display answer
                    st.markdown(answer)
                    st.caption(f"Response time: {response_time:.2f}s")
                    
                    # Display source documents
                    with st.expander(" View Source Documents"):
                        for i, doc in enumerate(source_docs):
                            st.write(f"**Document {i+1}:**")
                            st.write(doc.page_content)
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.caption(f"Source: {doc.metadata}")
                            st.markdown("---")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": source_docs
                    })
                    
                except Exception as e:
                    error_msg = f" Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
