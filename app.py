# app.py
import streamlit as st
import os
import faiss
import numpy as np
import nbformat
from typing import List

# --- New Imports for Gemini API ---
# We remove 'transformers' and 'torch'
from google import genai
from google.genai.errors import APIError 
# ---

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


# -------------------------
# Configuration / constants
# -------------------------
# NOTEBOOK_PATH is now unused but kept for reference if needed later
NOTEBOOK_PATH = "/mnt/data/RAG_practical.ipynb"  # server-side notebook path (provided)

# --- Updated LLM and API Config ---
GEMINI_MODEL_NAME = "gemini-2.5-flash" 
# Reads API key from environment variable (or Streamlit secrets)
# NOTE: Replace with a secure method in production
API_KEY = "AIzaSyBWcTBjqIyHtLSriSke6wvw20GeVIZfSRY"
# ----------------------------------

# Chunking params
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 4

# Generation params
MAX_NEW_TOKENS = 200 # Note: This is less relevant for API calls but kept for clarity
TEMPERATURE = 0.6    # Increased temperature to reduce repetition
TOP_P = 0.9

# -------------------------
# Helpers: text extraction
# -------------------------
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        try:
            ptext = page.extract_text()
        except Exception:
            ptext = None
        if ptext:
            pages.append(ptext)
    return "\n".join(pages)

# The extract_text_from_notebook function is now unused but kept for completeness
def extract_text_from_notebook(path: str) -> str:
    """Read local notebook and return concatenated markdown+code text."""
    if not os.path.exists(path):
        return ""
    nb = nbformat.read(path, as_version=4)
    parts = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source)
        elif cell.cell_type == "code":
            # Limiting code cell text to prevent huge chunks
            parts.append("\n".join(cell.source.splitlines()[:40]))
    return "\n\n".join(parts)

def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)

# -------------------------
# Helpers: chunking & indexing
# -------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

@st.cache_data(show_spinner=False)
def build_faiss_index(chunks: List[str], embedder_model_name="all-MiniLM-L6-v2"):
    if not chunks:
        return None, None, None
    embedder = SentenceTransformer(embedder_model_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index, embeddings, embedder

def retrieve_top_k(index, embedder: SentenceTransformer, chunks: List[str], question: str, top_k: int = TOP_K) -> List[str]:
    if index is None or not chunks:
        return []
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)
    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved.append(chunks[idx])
    return retrieved

# -------------------------
# Model loading (Gemini Client)
# -------------------------
@st.cache_resource
def load_gemini_client_and_embedder(api_key=API_KEY):
    if not api_key:
        # In a real app, this should probably be pulled from st.secrets
        raise ValueError("API_KEY not set. Cannot initialize Gemini Client.")
        
    # 1. Initialize Gemini Client
    client = genai.Client(api_key=api_key) 
    
    # 2. Initialize the SentenceTransformer for retrieval embeddings
    retrieval_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    return client, retrieval_embedder

# -------------------------
# Prompt / generation helper
# -------------------------
def make_prompt(context: str, question: str) -> str:
    prompt = (
        "You are a professional and factual Q&A assistant. Your goal is to answer the user's question "
        "using **ONLY** the text provided in the <CONTEXT> section below. \n\n"
        
        "## Instructions\n"
        "1. **STRICTLY USE CONTEXT:** Do not use any external knowledge, assumptions, or prior training data.\n"
        "2. **CONCISENESS:** Keep your answer as concise as possible while still being complete.\n"
        "3. **HANDLE UNCERTAINTY:** If the provided context is not sufficient to answer the question, "
        "you must respond with the phrase: **'I cannot find the answer in the provided documents.'**\n\n"
        
        "<CONTEXT>\n"
        f"{context}\n"
        "</CONTEXT>\n\n"
        
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt

def generate_answer(client, prompt: str, model_name=GEMINI_MODEL_NAME, temperature=TEMPERATURE, top_p=TOP_P):
    # Configure generation parameters
    config = {
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        # Call the Gemini API to generate the response
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        # Gemini handles the truncation and token limits internally
        return response.text.strip()
        
    except APIError as e:
        return f"Gemini API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred during generation: {e}"


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PDF/Text RAG Q&A", layout="wide")
st.title("🚀 PDF / Text RAG Q&A ")

left, right = st.columns([2, 1])

with left:
    st.header("1) Choose source")
    
    # --- MODIFICATION 1: Remove "Use uploaded notebook (server)" option ---
    source = st.radio("Select input type:", ("Upload PDF", "Paste text"))

    uploaded_pdf = None
    raw_text = ""

    if source == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"], key="pdf_uploader")
        if uploaded_pdf:
            st.info("PDF uploaded. It will be parsed and indexed.")
    else: # source == "Paste text"
        raw_text = st.text_area("Paste or type text here", height=200)

    st.markdown("---")
    st.header("2) Indexing options")
    chunk_size = st.number_input("Chunk size (words)", min_value=50, max_value=800, value=CHUNK_SIZE)
    overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=chunk_size-1, value=CHUNK_OVERLAP)
    top_k = st.number_input("Top-K retrieval chunks", min_value=1, max_value=10, value=TOP_K)

    st.markdown("---")
    st.header("3) Load Client & build index")
    if st.button("Initialize Client and build index"):
        
        # --- NEW MODEL LOADING ---
        with st.spinner(f"Initializing Client and loading embedding model..."):
            try:
                # Load the API client and the embedding model
                client, global_retrieval_embedder = load_gemini_client_and_embedder()
                st.success("Gemini Client and Embedding Model loaded.")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
                st.stop()
        # -------------------------

        if source == "Upload PDF" and uploaded_pdf:
            with st.spinner("Extracting text from PDF..."):
                txt = extract_text_from_pdf(uploaded_pdf)
                raw_text = clean_text(txt)
        # Note: Logic for 'Use uploaded notebook' is now gone
        
        if not raw_text.strip():
            st.warning("No text to index. Upload a PDF or paste text.")
            st.stop()

        chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        st.write(f"Created {len(chunks)} chunks (first chunk preview):")
        if chunks:
            st.text_area("Chunk[0] preview", chunks[0][:2000], height=150)

        with st.spinner("Building FAISS index (embedding chunks)..."):
            # Use the global_retrieval_embedder for index building
            index, embeddings, used_embedder = build_faiss_index(chunks, embedder_model_name="all-MiniLM-L6-v2")
            if index is None:
                st.error("Failed to build index (no chunks).")
                st.stop()
            st.success("FAISS index built and cached for this document.")

        st.session_state["ragg_state"] = {
            "chunks": chunks,
            "index": index,
            "embedder": used_embedder,
            "client": client, # Store the Gemini client
        }

with right:
    st.header("Ask questions")
    if "ragg_state" not in st.session_state:
        st.info("Initialize the Client and build index first.")
    else:
        question = st.text_input("Enter your question about the loaded document:")
        if st.button("Retrieve & Answer") and question.strip():
            state = st.session_state["ragg_state"]
            chunks = state["chunks"]
            index = state["index"]
            embedder = state["embedder"]
            client = state["client"] # Retrieve the Gemini client

            with st.spinner("Retrieving relevant chunks..."):
                retrieved = retrieve_top_k(index, embedder, chunks, question, top_k=top_k)
            
            if not retrieved:
                st.warning("No relevant chunks retrieved.")
                st.stop()

            # --- MODIFICATION 2: Remove display of retrieved chunks ---
            # st.subheader("Retrieved context (top chunks)")
            # for i, chunk in enumerate(retrieved, start=1):
            #     st.write(f"--- Chunk {i} ---")
            #     st.write(chunk)
            
            # Continue with generation
            context = "\n\n".join(retrieved)
            prompt = make_prompt(context, question)

            # --- NEW GENERATION CALL ---
            with st.spinner(f"Generating answer ({GEMINI_MODEL_NAME})..."):
                try:
                    answer = generate_answer(client, prompt=prompt, temperature=TEMPERATURE)
                    st.subheader("Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
            # ---------------------------

st.markdown("---")
st.caption("Forget hallucination. We built an AI Q&A system grounded entirely in your uploaded data.")