import os
import re
import csv
import uuid
import json
import time
from datetime import datetime
from collections import Counter
from math import log

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Robust PDF ingestion fallbacks
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ------------------- PATHS ------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_log.csv")

DOCS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../docs"))
PERSIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "../chroma_db"))
COLLECTION_NAME = "milestone1_docs"

BM25_STORE = os.path.join(PERSIST_DIR, "bm25_store.json")


# ------------------- UI THEME ------------------- #
def inject_css():
    st.markdown(
        """
        <style>
          /* ✅ Keep header container so sidebar toggle exists.
             Previously: display:none removed the toggle. */
          [data-testid="stHeader"] {
            background: transparent !important;
            height: 0px !important;
          }

          /* Hide toolbar content but keep container (don't break sidebar toggle) */
          [data-testid="stToolbar"] {
            visibility: hidden !important;
            height: 0px !important;
          }

          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }

          /* Layout spacing */
          .block-container { padding-top: 1.1rem; padding-bottom: 1.6rem; }

          .stApp {
            background: radial-gradient(1200px 600px at 20% 0%, rgba(99,102,241,.10), transparent 60%),
                        radial-gradient(1000px 500px at 100% 0%, rgba(16,185,129,.08), transparent 55%),
                        linear-gradient(180deg, rgba(2,6,23,.02), rgba(2,6,23,.00));
          }

          .rag-header {
            width: 100%;
            padding: 14px 16px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.18);
            background: rgba(255,255,255,.06);
            backdrop-filter: blur(10px);
            box-shadow: 0 12px 30px rgba(2,6,23,.18);
            margin: 0 0 14px 0;
          }
          .rag-title { font-size: 1.35rem; font-weight: 800; margin: 0; line-height: 1.2; color: rgba(255,255,255,.92); }
          .rag-sub { margin: .35rem 0 0 0; color: rgba(255,255,255,.62); font-size: .98rem; }

          /* Chat message cards (less gap between messages) */
          [data-testid="stChatMessage"] {
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,.14);
            background: rgba(255,255,255,.05);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 22px rgba(2,6,23,.20);
            margin-bottom: 8px;
          }
          [data-testid="stChatMessage"] * { color: rgba(255,255,255,.90) !important; }

          /* Sidebar */
          [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148,163,184,.14);
            background: rgba(2,6,23,.55);
            backdrop-filter: blur(12px);
          }
          [data-testid="stSidebar"] * { color: rgba(255,255,255,.90) !important; }

          /* Global buttons */
          .stButton>button {
            width: auto !important;
            border-radius: 12px !important;
            padding: 7px 12px !important;
            border: 1px solid rgba(148,163,184,.22) !important;
            background: rgba(255,255,255,.06) !important;
            box-shadow: 0 8px 18px rgba(2,6,23,.20) !important;
            white-space: nowrap !important;
            text-align: center !important;
            line-height: 1.1 !important;
            color: rgba(255,255,255,.92) !important;
          }
          .stButton>button:hover {
            transform: translateY(-1px);
            border-color: rgba(99,102,241,.55) !important;
          }

          /* Sidebar buttons full width */
          [data-testid="stSidebar"] .stButton>button {
            width: 100% !important;
            justify-content: center !important;
            font-weight: 650 !important;
          }

          /* Chat list wrap */
          .chat-list-wrap {
            max-height: 35vh;
            overflow: auto;
            padding-right: 4px;
          }

          /* Radio list -> ChatGPT-like items */
          [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 12px !important;
            padding: 10px 10px !important;
            margin: 6px 0 !important;
            border: 1px solid rgba(148,163,184,.14) !important;
            background: rgba(255,255,255,.05) !important;
            transition: all .12s ease-in-out;
          }
          [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: rgba(99,102,241,.35) !important;
            background: rgba(255,255,255,.08) !important;
            transform: translateY(-1px);
          }
          [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"]{
            transform: scale(0.01);
            opacity: 0;
            width: 0;
            height: 0;
            margin: 0;
            padding: 0;
          }
          [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            border-color: rgba(99,102,241,.55) !important;
            background: rgba(99,102,241,.14) !important;
            font-weight: 650 !important;
          }

          /* Message action buttons */
          .msg-actions {
            display:flex;
            gap:10px;
            align-items:center;
          }
          .msg-btn {
            border-radius: 10px;
            padding: 7px 12px;
            border: 1px solid rgba(148,163,184,.22);
            background: rgba(255,255,255,.06);
            cursor: pointer;
            font-size: 14px;
            color: rgba(255,255,255,.92);
            box-shadow: 0 8px 18px rgba(2,6,23,.20);
          }
          .msg-btn:hover {
            transform: translateY(-1px);
            border-color: rgba(99,102,241,.55);
          }
          .msg-toast {
            font-size: 12px;
            color: rgba(255,255,255,.60);
            margin-left: 4px;
          }

          /* Feedback buttons same size as msg buttons */
          .fb-wrap .stButton>button{
            border-radius: 10px !important;
            padding: 7px 12px !important;
            min-height: 34px !important;
          }

          /* Expander (sources) */
          [data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid rgba(148,163,184,.14);
            background: rgba(255,255,255,.04);
          }
          [data-testid="stExpander"] * { color: rgba(255,255,255,.90) !important; }

          /* Small headings */
          .sidebar-title { font-size: 1.05rem; font-weight: 800; margin: 0 0 .15rem 0; }
          .sidebar-sub { font-size: .85rem; color: rgba(255,255,255,.60); margin: 0 0 .55rem 0; }

          /* Edit question UI */
          .edit-row { margin-top: 8px; }
          .edit-actions {
            display:flex;
            gap:10px;
            align-items:center;
            margin-top: 10px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------- FEEDBACK LOGGER ------------------- #
def log_feedback(query, answer, feedback):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, query[:100], answer[:200], feedback]
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "answer", "feedback"])
        writer.writerow(row)


# ------------------- BM25 (LIGHTWEIGHT) ------------------- #
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str):
    return _TOKEN_RE.findall((text or "").lower())


def bm25_build(docs):
    k1 = 1.5
    b = 0.75

    freqs = []
    doc_len = []
    df = Counter()

    for d in docs:
        toks = _tokenize(d.get("text", ""))
        c = Counter(toks)
        freqs.append(c)
        dl = len(toks)
        doc_len.append(dl)
        for t in c.keys():
            df[t] += 1

    N = max(len(docs), 1)
    avgdl = (sum(doc_len) / N) if N else 0.0

    idf = {}
    for term, n_q in df.items():
        idf[term] = log(1 + (N - n_q + 0.5) / (n_q + 0.5))

    return {
        "k1": k1,
        "b": b,
        "N": N,
        "avgdl": avgdl,
        "docs": docs,
        "freqs": freqs,
        "doc_len": doc_len,
        "idf": idf,
    }


def bm25_search(index, query: str, k: int = 5):
    if not index or not index.get("docs"):
        return []
    q_terms = _tokenize(query)
    if not q_terms:
        return []

    k1 = index["k1"]
    b = index["b"]
    avgdl = index["avgdl"] or 1.0

    scores = []
    for i, c in enumerate(index["freqs"]):
        dl = index["doc_len"][i] or 0
        denom_norm = k1 * (1 - b + b * (dl / avgdl))
        s = 0.0
        for term in q_terms:
            if term not in c:
                continue
            tf = c[term]
            idf = index["idf"].get(term, 0.0)
            s += idf * (tf * (k1 + 1)) / (tf + denom_norm)
        scores.append((s, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = [index["docs"][i] for (s, i) in scores[:k] if s > 0]
    return top


def bm25_load_store():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    if not os.path.isfile(BM25_STORE):
        return []
    try:
        with open(BM25_STORE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("docs", []) if isinstance(data, dict) else []
    except Exception:
        return []


def bm25_save_store(docs):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    payload = {"docs": docs}
    with open(BM25_STORE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def ensure_bm25_ready():
    if "bm25_docs" not in st.session_state:
        st.session_state.bm25_docs = bm25_load_store()
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = bm25_build(st.session_state.bm25_docs)


def bm25_add_documents(chunks):
    ensure_bm25_ready()

    new_docs = []
    for d in chunks:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        md = dict(d.metadata or {})
        new_docs.append({"text": txt, "metadata": md})

    if not new_docs:
        return

    st.session_state.bm25_docs.extend(new_docs)
    bm25_save_store(st.session_state.bm25_docs)
    st.session_state.bm25_index = bm25_build(st.session_state.bm25_docs)


# ------------------- LOAD RAG ------------------- #
@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # NOTE: we keep retriever, but we will FILTER manually during retrieve
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a document QA assistant.

Rules:
- Use ONLY the provided document excerpts as evidence.
- If the answer is not in the excerpts, say: "I couldn't find that in the uploaded documents."
- Keep the answer concise.

Documents:
{context}

Question: {question}

Answer:"""
    )

    qa_chain = prompt | llm
    return qa_chain, retriever, vectorstore, embeddings


# ------------------- PDF -> CHROMA ------------------- #
def add_pdf_to_chroma(pdf_path: str, vectorstore, batch_size: int = 64):
    docs = None
    used = None

    try:
        docs = PyPDFLoader(pdf_path).load()
        used = "PyPDFLoader"
    except Exception:
        try:
            docs = PDFMinerLoader(pdf_path).load()
            used = "PDFMinerLoader"
        except Exception:
            try:
                import fitz  # pymupdf
                from langchain_core.documents import Document

                docs = []
                with fitz.open(pdf_path) as doc:
                    for i, page in enumerate(doc):
                        text = page.get_text("text") or ""
                        docs.append(Document(page_content=text, metadata={"source": pdf_path, "page": i}))
                used = "PyMuPDF (fitz)"
            except Exception:
                raise RuntimeError(
                    "Could not load this PDF using PyPDFLoader or PDFMinerLoader. "
                    "If you want a stronger fallback, add 'pymupdf' to requirements.txt."
                )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = splitter.split_documents(docs)

    for d in chunks:
        d.metadata["source"] = pdf_path  # IMPORTANT

    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])

    bm25_add_documents(chunks)

    return len(docs), len(chunks), used


# ------------------- MESSAGE ACTIONS (Copy + Voice) ------------------- #
def message_actions(text: str, key: str):
    safe = json.dumps(text)
    html = f"""
    <div class="msg-actions">
      <button class="msg-btn" id="{key}_copy">📋 Copy</button>
      <button class="msg-btn" id="{key}_read">🔊 Read</button>
      <button class="msg-btn" id="{key}_stop">✋ Stop</button>
      <span class="msg-toast" id="{key}_toast"></span>
    </div>

    <script>
      const txt = {safe};
      const toast = document.getElementById("{key}_toast");
      const copyBtn = document.getElementById("{key}_copy");
      const readBtn = document.getElementById("{key}_read");
      const stopBtn = document.getElementById("{key}_stop");

      function showToast(msg) {{
        toast.textContent = msg;
        setTimeout(() => {{ toast.textContent = ""; }}, 1200);
      }}

      copyBtn.onclick = async () => {{
        try {{
          await navigator.clipboard.writeText(txt);
          showToast("Copied");
        }} catch (e) {{
          try {{
            const ta = document.createElement("textarea");
            ta.value = txt;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            showToast("Copied");
          }} catch (e2) {{
            showToast("Copy failed");
          }}
        }}
      }};

      readBtn.onclick = () => {{
        try {{
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(txt);
          window.speechSynthesis.speak(u);
        }} catch (e) {{}}
      }};

      stopBtn.onclick = () => {{
        try {{ window.speechSynthesis.cancel(); }} catch (e) {{}}
      }};
    </script>
    """
    st.components.v1.html(html, height=44)


# ------------------- CHAT STATE ------------------- #
def init_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = create_chat()
    if "editing_msg_id" not in st.session_state:
        st.session_state.editing_msg_id = None
    if "edit_text" not in st.session_state:
        st.session_state.edit_text = ""
    # ✅ track current PDF
    if "active_pdf" not in st.session_state:
        st.session_state.active_pdf = None


def create_chat():
    cid = str(uuid.uuid4())[:8]
    st.session_state.chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "createdAt": datetime.now().timestamp(),
    }
    return cid


def delete_chat(chat_id: str):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    if not st.session_state.chats:
        st.session_state.active_chat_id = create_chat()
    else:
        items = list(st.session_state.chats.items())
        items.sort(key=lambda x: x[1]["createdAt"], reverse=True)
        st.session_state.active_chat_id = items[0][0]


def set_title_from_first_user(chat_id: str, user_text: str):
    chat = st.session_state.chats[chat_id]
    if chat["title"] == "New Chat":
        t = user_text.strip()
        t = t[:40] + ("..." if len(t) > 40 else "")
        chat["title"] = t if t else "New Chat"


def pretty_source_from_langchain(doc):
    src = doc.metadata.get("source", "Unknown")
    filename = os.path.basename(src) if src else "Unknown"
    page = doc.metadata.get("page", None)
    try:
        page_txt = "N/A" if page is None else str(int(page) + 1)
    except Exception:
        page_txt = str(page)
    return f"Page {page_txt}: **{filename}**"


def pretty_source_from_bm25(meta):
    src = meta.get("source", "Unknown")
    filename = os.path.basename(src) if src else "Unknown"
    page = meta.get("page", None)
    try:
        page_txt = "N/A" if page is None else str(int(page) + 1)
    except Exception:
        page_txt = str(page)
    return f"Page {page_txt}: **{filename}**"


# ✅ ONLY change: restrict retrieval to current active_pdf
def hybrid_retrieve(query: str, retriever, vectorstore, k_vec: int = 3, k_bm25: int = 5, k_final: int = 4):
    """
    Returns:
      context_texts: list[str]
      sources: list[str]
    """
    ensure_bm25_ready()

    active_pdf = st.session_state.get("active_pdf")

    # ---------- VECTOR (FILTERED) ----------
    if active_pdf:
        try:
            vec_docs = vectorstore.similarity_search(query, k=k_vec, filter={"source": active_pdf})
        except Exception:
            # fallback if filter unsupported in your langchain-chroma version
            all_vec = retriever.invoke(query)
            vec_docs = [d for d in all_vec if (d.metadata or {}).get("source") == active_pdf][:k_vec]
    else:
        vec_docs = retriever.invoke(query)

    # ---------- BM25 (FILTERED) ----------
    bm_docs = bm25_search(st.session_state.bm25_index, query, k=k_bm25)
    if active_pdf:
        bm_docs = [d for d in bm_docs if (d.get("metadata") or {}).get("source") == active_pdf]

    merged = []
    seen = set()

    def add_item(text, meta, source_pretty):
        sig = (str(meta.get("source", "")), str(meta.get("page", "")), (text[:120] if text else ""))
        if sig in seen:
            return
        seen.add(sig)
        merged.append((text, source_pretty))

    # Prefer vector first
    for d in vec_docs:
        txt = (d.page_content or "").strip()
        add_item(txt, d.metadata or {}, pretty_source_from_langchain(d))

    for d in bm_docs:
        txt = (d.get("text") or "").strip()
        meta = d.get("metadata") or {}
        add_item(txt, meta, pretty_source_from_bm25(meta))

    merged = merged[:k_final]
    context_texts = [m[0] for m in merged if m[0]]
    sources = [m[1] for m in merged if m[1]]
    return context_texts, sources


# ✅ GREETING HANDLING (added)
_GREET_RE = re.compile(
    r"^\s*(hi|hey|hello|hlo|hai|hola|yo|sup|good\s*(morning|afternoon|evening|night)|gm|ga|ge)\b[!.?,\s]*$",
    re.IGNORECASE,
)

def _is_greeting(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) > 40:  # avoid matching long sentences
        return False
    return _GREET_RE.match(t) is not None


def run_rag_and_append_answer(qa_chain, retriever, vectorstore, query: str, active_chat: dict):
    # ✅ GREETING HANDLING (added) - reply like ChatGPT without touching prompt/RAG
    if _is_greeting(query):
        answer_text = "Hi 👋 How can I help you today?\n\nUpload a PDF in the sidebar (if you want), then ask me anything about it."
        active_chat["messages"].append(
            {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": answer_text,
                "sources": [],
                "query": query,
                "feedback": None,
                "retrieval_time": 0.0,
                "dev_context": [],
            }
        )
        return

    # If no active pdf -> instruct user (safe)
    if not st.session_state.get("active_pdf"):
        answer_text = "👋 Please upload a PDF from the sidebar first. Then ask questions about that PDF."
        active_chat["messages"].append(
            {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": answer_text,
                "sources": [],
                "query": query,
                "feedback": None,
                "retrieval_time": 0.0,
                "dev_context": [],
            }
        )
        return

    t0 = time.time()
    context_texts, sources = hybrid_retrieve(query, retriever=retriever, vectorstore=vectorstore)
    retrieval_time = round(time.time() - t0, 2)

    context_joined = " ".join(context_texts) if context_texts else ""
    if (not context_texts) or (len(context_joined) < 150):
        answer_text = (
            "⚠️ I couldn't find reliable information about this in the **current uploaded PDF**.\n\n"
            "Try rephrasing your question or upload a more relevant PDF."
        )
    else:
        context_blob = "\n\n---\n\n".join(context_texts)
        answer = qa_chain.invoke({"context": context_blob, "question": query})
        answer_text = answer.content if hasattr(answer, "content") else str(answer)

    active_chat["messages"].append(
        {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer_text,
            "sources": sources,  # ✅ now only current PDF sources
            "query": query,
            "feedback": None,
            "retrieval_time": retrieval_time,
            "dev_context": context_texts,
        }
    )


# ------------------- APP ------------------- #
# IMPORTANT: set_page_config must be the FIRST Streamlit call
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
init_state()
ensure_bm25_ready()

qa_chain, retriever, vectorstore, embeddings = load_components()

# Top header + Download button
top_left, top_right = st.columns([7, 2], vertical_alignment="center")
with top_left:
    st.markdown(
        """
        <div class="rag-header">
          <p class="rag-title">🔍 Document RAG Assistant</p>
          <p class="rag-sub">Precise retrieval from your PDFs with verified sources</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    active_chat_id = st.session_state.active_chat_id
    active_chat = st.session_state.chats[active_chat_id]
    chat_export = {
        "chat_id": active_chat_id,
        "title": active_chat.get("title", "New Chat"),
        "messages": active_chat.get("messages", []),
    }
    st.download_button(
        "⬇️ Download chat",
        data=json.dumps(chat_export, indent=2),
        file_name=f"{active_chat.get('title','chat').replace(' ','_')}.json",
        mime="application/json",
        use_container_width=True,
    )

active_chat_id = st.session_state.active_chat_id
active_chat = st.session_state.chats[active_chat_id]

# Welcome message (2 lines)
if not active_chat.get("messages"):
    st.info("👋 Upload a PDF from the sidebar, then ask questions about it.")


# ------------------- SIDEBAR ------------------- #
with st.sidebar:
    st.markdown("## 🕘 History")

    dev_mode = st.toggle("🛠 Developer Mode", value=False)
    st.session_state.dev_mode = dev_mode

    if st.button("➕  New Chat", use_container_width=True):
        st.session_state.active_chat_id = create_chat()
        st.rerun()

    st.divider()

    st.markdown("### 📄 Upload PDF")
    up = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if up is not None:
        os.makedirs(DOCS_DIR, exist_ok=True)
        save_path = os.path.join(DOCS_DIR, up.name)
        with open(save_path, "wb") as f:
            f.write(up.getbuffer())

        try:
            with st.spinner("Indexing PDF into Chroma..."):
                pages, chunks, used = add_pdf_to_chroma(save_path, vectorstore, batch_size=64)
            # ✅ set active pdf to the latest uploaded
            st.session_state.active_pdf = save_path
            st.success(f"✅ Added {pages} pages, {chunks} chunks ({used})")
        except Exception as e:
            st.error("❌ Could not read/index this PDF. Try re-saving/exporting it, or upload a different file.")
            st.caption(f"Technical details: {type(e).__name__}: {e}")

    # ✅ show active pdf (so you know what it's using)
    if st.session_state.get("active_pdf"):
        st.caption(f"📌 Active PDF: **{os.path.basename(st.session_state.active_pdf)}**")

    st.divider()

    # Chats header + delete icon aligned
    h1, h2 = st.columns([6, 1])
    with h1:
        st.markdown('<p class="sidebar-title">Chats</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-sub">Your recent conversations</p>', unsafe_allow_html=True)
    with h2:
        delete_disabled = len(st.session_state.chats) <= 1
        if st.button("🗑", key="delete_selected_chat_icon", help="Delete selected chat", disabled=delete_disabled):
            delete_chat(st.session_state.active_chat_id)
            st.rerun()

    q = st.text_input("Search chats", placeholder="Search...", label_visibility="collapsed")

    chat_items = list(st.session_state.chats.items())
    chat_items.sort(key=lambda x: x[1]["createdAt"], reverse=True)

    if q.strip():
        q_low = q.strip().lower()
        chat_items = [(cid, c) for cid, c in chat_items if q_low in (c.get("title", "").lower())]

    chat_ids = [cid for cid, _ in chat_items]

    def chat_label(cid: str) -> str:
        title = st.session_state.chats[cid].get("title", "New Chat") or "New Chat"
        title = title.strip() or "New Chat"
        return (title[:28] + "…") if len(title) > 29 else title

    if not chat_ids:
        st.caption("No chats found.")
    else:
        st.markdown('<div class="chat-list-wrap">', unsafe_allow_html=True)

        selected = st.radio(
            label="Select chat",
            options=chat_ids,
            index=chat_ids.index(st.session_state.active_chat_id)
            if st.session_state.active_chat_id in chat_ids
            else 0,
            format_func=chat_label,
            label_visibility="collapsed",
            key="chat_selector_radio",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if selected != st.session_state.active_chat_id:
            st.session_state.active_chat_id = selected
            st.rerun()


# ------------------- EDIT QUESTION (Perplexity-like) ------------------- #
def start_edit(msg_id: str, current_text: str):
    st.session_state.editing_msg_id = msg_id
    st.session_state.edit_text = current_text


def cancel_edit():
    st.session_state.editing_msg_id = None
    st.session_state.edit_text = ""


def save_and_regenerate(active_chat: dict, msg_id: str, new_text: str):
    idx = None
    for i, m in enumerate(active_chat["messages"]):
        if m.get("id") == msg_id and m.get("role") == "user":
            idx = i
            break
    if idx is None:
        cancel_edit()
        return

    active_chat["messages"][idx]["content"] = new_text

    if idx + 1 < len(active_chat["messages"]) and active_chat["messages"][idx + 1].get("role") == "assistant":
        active_chat["messages"].pop(idx + 1)

    run_rag_and_append_answer(qa_chain, retriever, vectorstore, new_text, active_chat)

    cancel_edit()


# ------------------- RENDER MESSAGES ------------------- #
for m in active_chat["messages"]:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            head = st.columns([12, 1], vertical_alignment="center")
            with head[0]:
                st.markdown(m["content"])
            with head[1]:
                if st.button("✏️", key=f"edit_btn_{m['id']}", help="Edit question"):
                    start_edit(m["id"], m["content"])
                    st.rerun()

            if st.session_state.editing_msg_id == m["id"]:
                st.markdown('<div class="edit-row">', unsafe_allow_html=True)
                new_text = st.text_area(
                    "Edit your question",
                    value=st.session_state.edit_text,
                    key=f"edit_area_{m['id']}",
                    height=120,
                    label_visibility="collapsed",
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="edit-actions">', unsafe_allow_html=True)
                b1, b2 = st.columns([3, 2], vertical_alignment="center")
                with b1:
                    if st.button("✅ Save & Regenerate", key=f"save_regen_{m['id']}"):
                        save_and_regenerate(active_chat, m["id"], new_text.strip())
                        st.rerun()
                with b2:
                    if st.button("✖ Cancel", key=f"cancel_edit_{m['id']}"):
                        cancel_edit()
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            continue

        st.markdown(m["content"])

        if "retrieval_time" in m:
            st.caption(f"⏱ Retrieved in {m['retrieval_time']} seconds")

        action_col, fb_col = st.columns([7, 2], vertical_alignment="center")
        with action_col:
            message_actions(m["content"], key=f"act_{m['id']}")

        with fb_col:
            st.markdown('<div class="fb-wrap">', unsafe_allow_html=True)
            fb1, fb2 = st.columns(2)
            st.markdown("</div>", unsafe_allow_html=True)

            def set_feedback(val: str | None):
                m["feedback"] = val
                if val in ("up", "down"):
                    log_feedback(m.get("query", ""), m["content"], val)

            up_active = (m.get("feedback") == "up")
            down_active = (m.get("feedback") == "down")

            with fb1:
                if st.button(("👍" if not up_active else "👍✅"), key=f"up_{m['id']}", help="Helpful"):
                    set_feedback(None if up_active else "up")
                    st.rerun()

            with fb2:
                if st.button(("👎" if not down_active else "👎✅"), key=f"down_{m['id']}", help="Not helpful"):
                    set_feedback(None if down_active else "down")
                    st.rerun()

        if "sources" in m and m["sources"]:
            with st.expander(f"📄 Sources ({len(m['sources'])})", expanded=False):
                for i, src in enumerate(m["sources"], 1):
                    st.markdown(f"**{i}.** {src}")

        if st.session_state.get("dev_mode") and m.get("dev_context"):
            with st.expander("🛠 Debug: Retrieved Context", expanded=False):
                for i, chunk in enumerate(m["dev_context"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.code(chunk[:900])


# ------------------- INPUT + RAG ANSWER ------------------- #
query = st.chat_input("Ask about your documents...", key="rag_input")

if query:
    user_id = str(uuid.uuid4())
    active_chat["messages"].append({"id": user_id, "role": "user", "content": query})
    set_title_from_first_user(active_chat_id, query)

    with st.spinner("Searching documents..."):
        run_rag_and_append_answer(qa_chain, retriever, vectorstore, query, active_chat)

    st.rerun()