from dotenv import load_dotenv
from Utils.vector_db_handler import VectorDBHandler

# Load environment variables
load_dotenv()

# Define the directory paths and collection name
files_directory = "./resumes"
persist_directory = "./vector_db"
collection_name = "resumes_collection"

# Initialize the vector database handler
vector_db_handler = VectorDBHandler(files_directory, persist_directory, collection_name)

# Load or create the vector store database
vector_db_handler.load_or_create_db()

# Define the query for the search
query = "I am looking for a person with communication skills. who knows kubernetes."
try:
    docs = vector_db_handler.query_vector_store(query)
    
    # Output the top result
    if docs:
        print("Top matching document:")
        print(docs[0].page_content)
    else:
        print("No matching documents found.")
except ValueError as e:
    print(f"Error: {e}")
