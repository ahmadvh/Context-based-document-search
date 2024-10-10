import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4

class VectorDBHandler:
    def __init__(self, directory_path, persist_directory, collection_name):
        self.directory_path = directory_path
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store = None
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=512,
        )

    def check_if_db_exists(self):
        """Check if the vector database already exists in the specified directory."""
        return os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory)

    def create_vector_embedding(self):
        """Create vector embeddings for text files in the specified directory and store them in Chroma."""
        documents = []
        for file_name in os.listdir(self.directory_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.directory_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    metadata = {"file_path": file_path}
                    documents.append(Document(page_content=content, metadata=metadata))

        # Initialize the vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )

        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents, ids=uuids)
        print(f"Added {len(documents)} documents to the vector store.")

    def load_or_create_db(self):
        """Load the vector store if it exists; otherwise, create a new one."""
        if self.check_if_db_exists():
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
        else:
            print("No existing vector store found. Creating a new one...")
            self.create_vector_embedding()

    def query_vector_store(self, query_text, top_k=5):
        """Query the vector store for similar documents to the given query text."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Use load_or_create_db() first.")
        
        results = self.vector_store.similarity_search(query=query_text, k=top_k)
        
        return results
