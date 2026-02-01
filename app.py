import streamlit as st
import tempfile
import os
import re
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings


# ================= PAGE SETUP =================
st.set_page_config(page_title="AI Document Search using RAG", layout="centered")
st.title("📄 AI Document Search using RAG")

st.markdown("""
Upload documents and ask questions.  
Answers are generated **strictly from the uploaded documents**.
""")


# ================= SESSION STATE =================
for k, v in {
    "chat_history": [],
    "fill_query": "",
    "last_uploaded_files": None,
    "suggestions": []
}.items():
    st.session_state.setdefault(k, v)


# ================= FILE SAVE =================
def save_file(file):
    data = file.getvalue()
    if not data:
        return None
    suffix = "." + file.name.split(".")[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


# ================= CLEAR CHAT =================
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history.clear()
    st.session_state.fill_query = ""
    st.session_state.suggestions.clear()
    st.cache_resource.clear()
    st.rerun()


# ================= FILE UPLOAD =================
uploaded_files = st.file_uploader(
    "Upload document(s)",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

current_files = tuple(f.name for f in uploaded_files) if uploaded_files else None
if current_files != st.session_state.last_uploaded_files:
    st.session_state.chat_history.clear()
    st.session_state.fill_query = ""
    st.session_state.suggestions.clear()
    st.cache_resource.clear()
    st.session_state.last_uploaded_files = current_files


# ================= LOAD DOCUMENTS =================
def load_documents(files):
    documents = []

    for f in files:
        path = save_file(f)
        if not path:
            continue

        loader = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "csv": CSVLoader
        }.get(f.name.split(".")[-1].lower())

        if loader:
            try:
                documents.extend(loader(path).load())
            except Exception:
                pass

        os.remove(path)

    return documents


# ================= VECTOR DB (SAFE) =================
@st.cache_resource
def build_vector_db(files):
    docs = load_documents(files)
    if not docs:
        raise ValueError("No readable content found.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)
    return Chroma.from_documents(chunks, embeddings)


# ================= LLM =================
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


# ================= SUGGESTIONS =================
def generate_suggestions(files):
    templates = [
        "Explain the concept of {} in detail.",
        "How does {} work in practical applications?",
        "What are the main steps involved in {}?",
        "Why is {} important?"
    ]

    results = []
    for f in files:
        path = save_file(f)
        if not path:
            continue

        loader = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "csv": CSVLoader
        }.get(f.name.split(".")[-1].lower())

        if not loader:
            os.remove(path)
            continue

        try:
            text = " ".join(d.page_content for d in loader(path).load())
        except Exception:
            os.remove(path)
            continue

        words = re.findall(r"\b[a-zA-Z]{6,}\b", text.lower())
        keywords = [w for w, _ in Counter(words).most_common(3)]

        for kw in keywords:
            for t in templates:
                results.append({
                    "question": t.format(kw),
                    "source": f.name
                })

        os.remove(path)

    return results


# ================= PROCESS FILES =================
if uploaded_files:
    with st.spinner("Indexing documents..."):
        vector_db = build_vector_db(uploaded_files)
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    if not st.session_state.suggestions:
        st.session_state.suggestions = generate_suggestions(uploaded_files)


# ================= QUESTION INPUT =================
query = st.text_input(
    "Ask a question from the document:",
    value=st.session_state.fill_query,
    placeholder="Type a complete question"
)

col1, col2 = st.columns(2)
ask_clicked = col1.button("🔎 Ask")
if col2.button("🧹 Clear Question"):
    st.session_state.fill_query = ""
    st.rerun()

query = query.strip()


# ================= SHOW SUGGESTIONS =================
if uploaded_files and query == "":
    st.markdown("💡 **Suggested Questions (with source)**")
    for s in st.session_state.suggestions:
        c1, c2 = st.columns([4, 2])
        if c1.button(s["question"], key=s["question"] + s["source"]):
            st.session_state.fill_query = s["question"]
            st.rerun()
        c2.caption(f"📄 {s['source']}")


# ================= ANSWER =================
def ask_ai(docs, question):
    context = " ".join(d.page_content for d in docs)
    if len(context.strip()) < 30:
        return "Information not found in the uploaded documents."

    prompt = f"""
Answer using ONLY the context below.

FORMAT:
- One short paragraph
- 4 bullet points

Context:
{context}

Question:
{question}

Answer:
"""

    tok, model = load_llm()
    out = model.generate(**tok(prompt, return_tensors="pt", truncation=True), max_new_tokens=200)
    return tok.decode(out[0], skip_special_tokens=True)


# ================= RUN QUERY =================
if uploaded_files and ask_clicked and len(query.split()) >= 3:
    docs = retriever.invoke(query)
    answer = ask_ai(docs, query)

    st.markdown("### 🧠 Answer")
    st.write(answer)

    if "not found" not in answer.lower():
        st.session_state.chat_history.append((query, answer))


# ================= CHAT HISTORY =================
if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"### Q{i+1}: {q}")
        st.write(a)
