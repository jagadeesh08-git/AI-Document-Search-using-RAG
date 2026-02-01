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
Upload one or more documents and ask questions.  
Answers are generated **strictly from uploaded documents**.
""")


# ================= SESSION STATE =================
defaults = {
    "chat_history": [],
    "query": "",
    "last_files": None,
    "suggestions": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ================= FILE SAVE =================
def save_file(file):
    data = file.getvalue()
    if not data:
        return None
    suffix = "." + file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


# ================= LOAD DOCUMENTS =================
def load_docs(files):
    documents = []

    for file in files:
        path = save_file(file)
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
                for d in docs:
                    d.metadata["source"] = file.name
                documents.extend(docs)
            except Exception:
                pass

        os.remove(path)

    return documents


# ================= VECTOR DB =================
@st.cache_resource
def build_vector_db(files):
    docs = load_docs(files)
    if not docs:
        raise ValueError("No readable text found in uploaded documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text chunks created from documents.")

    embeddings = FakeEmbeddings(size=384)
    db = Chroma.from_documents(chunks, embeddings)
    return db, chunks


# ================= LLM =================
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


# ================= QUESTION TEMPLATES (11 TYPES) =================
QUESTION_TEMPLATES = [
    "Explain the concept of {} in detail.",
    "How does {} work in practical applications?",
    "What are the main steps involved in {}?",
    "Why is {} important in real-world systems?",
    "Discuss the advantages and limitations of {}.",
    "How is {} implemented and evaluated?",
    "Compare {} with related approaches or methods.",
    "What challenges are associated with {}?",
    "In which scenarios is {} most effective?",
    "How is the performance of {} measured?",
    "What are the major application areas of {}?"
]


# ================= SUGGESTIONS =================
def generate_suggestions(chunks, max_per_doc=4):
    suggestions = []
    seen = set()

    for d in chunks:
        source = d.metadata.get("source", "Unknown")
        text = d.page_content.lower()

        words = re.findall(r"\b[a-zA-Z]{6,}\b", text)
        keywords = [w for w, _ in Counter(words).most_common(3)]

        for kw in keywords:
            for tmpl in QUESTION_TEMPLATES:
                q = tmpl.format(kw)
                key = (q, source)
                if key not in seen:
                    seen.add(key)
                    suggestions.append({
                        "question": q,
                        "source": source
                    })

    return suggestions


# ================= CLEAR CHAT =================
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history.clear()
    st.session_state.query = ""
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
if current_files != st.session_state.last_files:
    st.session_state.chat_history.clear()
    st.session_state.query = ""
    st.session_state.suggestions.clear()
    st.cache_resource.clear()
    st.session_state.last_files = current_files


# ================= PROCESS FILES =================
if uploaded_files:
    with st.spinner("Indexing documents..."):
        try:
            vector_db, chunks = build_vector_db(uploaded_files)
            retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            st.error(str(e))
            st.stop()

    if not st.session_state.suggestions:
        st.session_state.suggestions = generate_suggestions(chunks)


# ================= QUESTION INPUT =================
st.session_state.query = st.text_input(
    "Ask a question from the document:",
    value=st.session_state.query,
    placeholder="Type a complete question"
)

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("🔎 Ask")
with col2:
    clear_q = st.button("🧹 Clear Question")

if clear_q:
    st.session_state.query = ""
    st.rerun()

query = st.session_state.query.strip()


# ================= SHOW SUGGESTIONS =================
if uploaded_files and st.session_state.suggestions and query == "":
    st.markdown("💡 **Suggested Questions (document-based)**")
    for s in st.session_state.suggestions:
        col_q, col_s = st.columns([4, 2])
        with col_q:
            if st.button(s["question"], key=s["question"] + s["source"]):
                st.session_state.query = s["question"]
                st.rerun()
        with col_s:
            st.caption(f"📄 {s['source']}")


# ================= ANSWER =================
def answer_question(docs, question):
    context = " ".join(d.page_content for d in docs)
    if len(context.strip()) < 30:
        return "Information not found in the uploaded documents."

    prompt = f"""
Answer using ONLY the context below.

FORMAT:
- One short paragraph
- Exactly 4 bullet points

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
    docs = retriever.invoke(query)
    if docs:
        answer = answer_question(docs, query)
        st.markdown("### 🧠 Answer")
        st.write(answer)

        if "not found" not in answer.lower():
            st.session_state.chat_history.append((query, answer))


# ================= CHAT HISTORY =================
if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}: {q}**")
        st.write(a)
