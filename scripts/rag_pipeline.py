import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = "../chroma_db"

def load_rag_pipeline():
    print("🔄 Loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    print("🔄 Loading LLM...")
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",  # ✅ Active Groq model (128k context)
        temperature=0.1
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the following context. 
        Cite sources using [page X from filename].
        
        Context: {context}
        
        Question: {question}
        
        Answer with citations:"""
    )
    
    print("✅ RAG ready!\n")
    return vectorstore.as_retriever(search_kwargs={"k": 4}), llm, prompt

def main():
    retriever, llm, prompt = load_rag_pipeline()
    
    while True:
        query = input("\nAsk about your PDFs (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
            
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        chain = prompt | llm
        answer = chain.invoke({"context": context, "question": query}).content
        
        sources = [f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} (page {doc.metadata.get('page', 'N/A')})" 
                  for doc in docs]
        print(f"\n🤖 {answer}")
        print(f"\n📄 Sources: {sources}")

if __name__ == "__main__":
    main()
