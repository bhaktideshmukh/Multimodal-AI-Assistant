from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_data():
    print("1. Loading Medical Reference Guide...")
    loader = TextLoader("medical_reference_guide.md")
    documents = loader.load()

    print("2. Splitting text into chunks...")
    # We cut the document into 500-character chunks so the LLM gets very specific context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    print("3. Initializing HuggingFace Embeddings (May take a minute to download)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("4. Saving to local ChromaDB...")
    # This creates a folder called 'chroma_db' in your project and saves the vectors there.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Ingestion Complete! Vector Database is ready.")

if __name__ == "__main__":
    ingest_data()
