from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "../chroma_db"

def main():
    print("🔄 Loading existing Chroma DB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="milestone1_docs",
    )
    print("✅ Vector store loaded.\n")

    while True:
        query = input("Enter your question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break

        results = vectorstore.similarity_search(query, k=3)
        print("\n✅ Top chunks:")
        for i, doc in enumerate(results, start=1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            print(f"\n[{i}] Source: {src} | page: {page}")
            print(doc.page_content[:400] + "...")

if __name__ == "__main__":
    main()
