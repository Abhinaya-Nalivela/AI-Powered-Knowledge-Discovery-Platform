import os
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "../chroma_db"
DOCS_DIR = "../docs"
COLLECTION_NAME = "milestone1_docs"


def safe_load_pdf(path: str):
    """
    Try multiple PDF parsers so ingestion doesn't crash.
    Returns (docs, used_loader_name) or (None, reason).
    """
    # 1) PyPDF
    try:
        docs = PyPDFLoader(path).load()
        return docs, "PyPDFLoader"
    except Exception as e1:
        # 2) PDFMiner fallback
        try:
            docs = PDFMinerLoader(path).load()
            return docs, "PDFMinerLoader"
        except Exception as e2:
            # 3) PyMuPDF fallback (works on many malformed PDFs)
            try:
                import fitz  # pymupdf
                texts = []
                with fitz.open(path) as doc:
                    for i, page in enumerate(doc):
                        t = page.get_text("text") or ""
                        # emulate LangChain Document
                        from langchain_core.documents import Document
                        texts.append(
                            Document(
                                page_content=t,
                                metadata={"source": path, "page": i}
                            )
                        )
                return texts, "PyMuPDF (fitz)"
            except Exception as e3:
                return None, f"PyPDF: {type(e1).__name__} | PDFMiner: {type(e2).__name__} | PyMuPDF: {type(e3).__name__}"


def load_all_pdfs():
    pdf_paths = []
    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, f))

    all_docs = []
    failed = []

    print("🔄 Loading PDFs...")
    for p in pdf_paths:
        docs, used = safe_load_pdf(p)
        if docs is None:
            failed.append((p, used))
            print(f"❌ Skipped: {p}\n   Reason: {used}")
            continue

        # ensure source metadata
        for d in docs:
            d.metadata["source"] = p

        all_docs.extend(docs)
        print(f"✅ Loaded: {os.path.basename(p)} ({used})")

    print(f"\n✅ Total loaded pages/docs: {len(all_docs)}")
    if failed:
        print(f"⚠️ Skipped {len(failed)} unreadable PDF(s).")

    return all_docs


def load_and_split():
    docs = load_all_pdfs()

    print("🔄 Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks!")
    return chunks


def build_vectorstore(chunks):
    print("🔄 Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"🔄 Creating Chroma DB at {PERSIST_DIR} ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    print("✅ Stored chunks in Chroma!")


def main():
    chunks = load_and_split()
    if not chunks:
        print("❌ No chunks created (no readable PDFs found).")
        return
    build_vectorstore(chunks)
    print("\n🎯 ingestion + indexing complete.")


if __name__ == "__main__":
    main()