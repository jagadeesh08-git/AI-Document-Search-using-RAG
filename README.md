📄 AI Document Search using RAG

An AI-powered document search and question-answering system built using the Retrieval-Augmented Generation (RAG) paradigm.
The application allows users to upload documents and ask questions, with answers generated strictly from the uploaded content.

🚀 Features

Upload multiple documents (PDF, TXT, CSV)

Document-based question answering (RAG)

Automatic document chunking and retrieval

AI-generated answers using a transformer model

Document-based question suggestions

Each suggested question shows its source document

Chat history with clear/reset options

Clean and interactive Streamlit UI

🧠 How It Works (RAG Pipeline)

Document Ingestion
Uploaded documents are loaded and split into smaller text chunks.

Retrieval
Relevant chunks are retrieved from a vector store based on the user query.

Augmentation
Retrieved content is injected into the prompt as context.

Generation
A transformer-based language model generates answers using only the provided context.

🗂️ Supported File Types

PDF

TXT

CSV

Note: Documents must contain machine-readable text. Scanned or empty files may be ignored.

🛠️ Tech Stack

Frontend: Streamlit

RAG Framework: LangChain

Vector Store: ChromaDB

Language Model: FLAN-T5 (Transformers)

Document Parsing: PyPDF, CSV Loader
## 📦 Installation

📦 Installation

Clone the repository:
https://ai-document-search-using-rag-6uatemasbkngub4q4x2jnc.streamlit.app/

pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py
