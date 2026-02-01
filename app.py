import streamlit as st
import tempfile
import os
import re
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader


# ================= PAGE SETUP =================
st.set_page_config(page_title="AI Document Search using RAG", layout="centered")
st.title("📄 AI Document Search using RAG")

st.markdown("""
Upload documents and ask questions.  
Answers are generated **strictly from the uploaded documents** using a lightweight RAG pipeline.
""")


# ================= SESSION STATE =================
defaults = {
    "documents": [],
    "chat_history": [],
    "fill_query": "",
    "suggestions": [],
    "last_uploaded_files": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ================= HELPERS =================
def save_uploaded_file(file):
    data = file.getvalue()
    if not data:
        return None

    suffix = "." + file.name.split(".")[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def load_documents(files):
    documents = []

    for file in files:
        path = save_uploaded_file(file)
        if not path:
            continue

        ext = file.name.split(".")[-1].lower()
        loader = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "csv": CSVLoader
        }.get(ext)

        if loader:
            try:
                docs = loader(path).load()
                documents.extend(docs)
            except Exception:
                pass

        os.remove(path)

    return documents


# ================= SIMPLE RETRIEVER (CLOUD SAFE) =================
def simple_retriever(docs, query, top_k=4):
    query_words = set(query.lower().split())
    scored = []

    for d in docs:
        text = d.page_content.lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, d))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:top_k]]


# ================= LLM =================
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


# ================= CLEAR CHAT =================
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history.clear()
    st.session_state.fill_query = ""
    st.session_state.suggestions.clear()
    st.success("Chat cleared")
    st.rerun()


# ================= FILE UPLOAD =================
uploaded_files = st.file_uploader(
    "Upload document(s)",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

current_files = tuple(f.name for f in uploaded_files) if uploaded_files else None
if current_files != st.session_state.last_uploaded_files:
    st.session_state.documents = []
    st.session_state.chat_history.clear()
    st.session_state.fill_query = ""
    st.session_state.suggestions.clear()
    st.session_state.last_uploaded_files = current_files


# ================= PROCESS FILES =================
if uploaded_files and not st.session_state.documents:
    with st.spinner("Processing documents..."):
        st.session_state.documents = load_documents(uploaded_files)

    if not st.session_state.documents:
        st.error("No readable text found in uploaded documents.")
        st.stop()


# ================= SUGGESTIONS (PER DOCUMENT) =================
def generate_suggestions(files, max_per_doc=3):
    QUESTION_TEMPLATES = [
        "Explain the concept of {} in detail.",
        "How does {} work in practical applications?",
        "What are the main steps involved in {}?",
        "Why is {} important?",
        "Discuss the advantages and limitations of {}."
    ]

    suggestions = []

    for file in files:
        path = save_uploaded_file(file)
        if not path:
            continue

        ext = file.name.split(".")[-1].lower()
        loader = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "csv": CSVLoader
        }.get(ext)

        if not loader:
            os.remove(path)
            continue

        try:
            docs = loader(path).load()
        except Exception:
            os.remove(path)
            continue

        text = " ".join(d.page_content for d in docs)
        words = re.findall(r"\b[a-zA-Z]{6,}\b", text.lower())
        keywords = [w for w, _ in Counter(words).most_common(4)]

        count = 0
        for kw in keywords:
            for tmpl in QUESTION_TEMPLATES:
                if count >= max_per_doc:
                    break
                suggestions.append({
                    "question": tmpl.format(kw),
                    "source": file.name
                })
                count += 1
            if count >= max_per_doc:
                break

        os.remove(path)

    return suggestions


if uploaded_files and not st.session_state.suggestions:
    st.session_state.suggestions = generate_suggestions(uploaded_files)


# ================= QUESTION INPUT =================
query = st.text_input(
    "Ask a question from the document:",
    value=st.session_state.fill_query,
    placeholder="Type a complete question"
)

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("🔎 Ask")
with col2:
    clear_query = st.button("🧹 Clear Question")

if clear_query:
    st.session_state.fill_query = ""
    st.rerun()

query = query.strip()


# ================= SHOW SUGGESTIONS =================
if uploaded_files and st.session_state.suggestions and query == "":
    st.markdown("💡 **Suggested Questions (with source)**")

    for s in st.session_state.suggestions:
        col_q, col_s = st.columns([4, 2])
        with col_q:
            if st.button(s["question"], key=s["question"] + s["source"]):
                st.session_state.fill_query = s["question"]
                st.rerun()
        with col_s:
            st.caption(f"📄 {s['source']}")


# ================= RAG ANSWER =================
def ask_ai(docs, question):
    context = " ".join(d.page_content for d in docs)

    if len(context.strip()) < 40:
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

    tokenizer, model = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ================= EXECUTE QUERY =================
if uploaded_files and ask_clicked and len(query.split()) >= 3:
    docs = simple_retriever(st.session_state.documents, query)

    answer = ask_ai(docs, query)

    st.markdown("### 🧠 Answer")
    st.write(answer)

    if "Information not found" not in answer:
        st.session_state.chat_history.append((query, answer))


# ================= CHAT HISTORY =================
if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"### Q{i+1}: {q}")
        st.write(a)
