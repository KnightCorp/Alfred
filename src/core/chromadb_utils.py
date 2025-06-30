import chromadb
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
import chromadb


def create_inmemory_collection():
    """
    Create an in-memory ChromaDB collection for storing documents.
    Returns the collection object.
    """
    client = chromadb.Client()
    collection = client.create_collection("docs")
    return collection

def add_file_to_collection(collection, file_path, file_id):
    """
    Reads the file at file_path and adds its content to the ChromaDB collection.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        collection.add(
            documents=[content],
            metadatas=[{"source": file_path}],
            ids=[file_id]
        )
        print(f"✅ Added file: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def query_collection(collection, query, n_results=3):
    """
    Queries the ChromaDB collection for the most relevant documents.
    Returns a string with the combined context.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results.get('documents', [[]])[0]
    if not docs:
        return ""
    context = "\n\n".join(docs)
    return context
