import os
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "../docs"


def safe_load_pdf(path: str):
    try:
        return PyPDFLoader(path).load(), "PyPDFLoader"
    except Exception:
        try:
            return PDFMinerLoader(path).load(), "PDFMinerLoader"
        except Exception:
            import fitz
            from langchain_core.documents import Document
            out = []
            with fitz.open(path) as doc:
                for i, page in enumerate(doc):
                    out.append(Document(page_content=page.get_text("text") or "", metadata={"source": path, "page": i}))
            return out, "PyMuPDF (fitz)"


def main():
    pdfs = []
    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))

    if not pdfs:
        print("No PDFs found.")
        return

    # test first PDF
    path = pdfs[0]
    print("🔄 Loading:", path)

    docs, used = safe_load_pdf(path)
    print(f"✅ Loaded {len(docs)} pages using {used}")
    print("Preview:", (docs[0].page_content[:200] if docs else "") + "...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    splits = splitter.split_documents(docs)
    print(f"✅ Split into {len(splits)} chunks!")
    if splits:
        print("First chunk:", splits[0].page_content[:200] + "...")
        print("Sample metadata:", splits[0].metadata)


if __name__ == "__main__":
    main()