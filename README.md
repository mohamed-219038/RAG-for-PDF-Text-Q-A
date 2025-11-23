# RAG-for-PDF-Text-Q-A
# 🚀 Gemini Flash-Powered RAG Q&A App

This repository contains a modern **Retrieval-Augmented Generation (RAG)** application built using **Streamlit** for the front-end interface, **FAISS** for vector indexing, and Google's powerful **Gemini 2.5 Flash** model for context-aware answer generation.

The application allows users to upload a PDF or paste raw text, index the content, and then ask questions that are answered *only* using the loaded document, providing highly factual and grounded results.

## ✨ Key Features

* **Gemini 2.5 Flash Integration:** Leverages the speed and strong reasoning capabilities of the Gemini Flash model for factual answer generation.
* **Streamlit UI:** Provides an easy-to-use, interactive web interface for document upload and Q&A.
* **Efficient Vector Search (FAISS):** Uses FAISS with `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`) for lightning-fast retrieval of the most relevant text chunks.
* **Grounded Answers:** Strict prompt engineering forces the model to use **ONLY** the retrieved context, ensuring high factual accuracy.
* **Flexible Input:** Supports text extraction from **PDF uploads** or direct **text pasting**.

## 🛠️ Setup and Installation

Follow these steps to get the application running locally.

### 1. Prerequisites

You need Python 3.9+ installed on your system.

