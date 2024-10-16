# Contextual Documents Search

This project provides a system for performing context-based search across documents stored in a vector database. Using OpenAI's embedding models and Chroma, this tool allows you to efficiently search through a collection of text documents and retrieve the most relevant results based on a given query.

## Features
- Automatic vector embedding generation for documents stored in a specified directory.
- Easy-to-use search functionality that finds the most contextually relevant documents.
- Persistent vector storage using Chroma, allowing for seamless loading and updating of the database.

## Prerequisites
- Python 3.7 or higher
- OpenAI API key
- Install the required packages by running:
  ```bash
  pip install -r requirements.txt
  ```

  ## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/contextual-documents-search.git
   ```
2. Navigate to the project directory:
   ```bash
   cd contextual-documents-search
   ```
3. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Set up your environment variables. Create a .env file in the project root and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY = your_openai_api_key
   ```
## Usage

### Initializing and Querying the Vector Database

1. Prepare a directory of `.txt` files you want to search through and place them in the `./resumes` folder or specify a different directory in the code.
2. In your main script, instantiate the `VectorDBHandler` class and call `load_or_create_db()` to initialize the vector store.

   ```python
   from dotenv import load_dotenv
   from vector_db_handler import VectorDBHandler

   # Load environment variables
   load_dotenv()

   # Set up directory paths and collection name
   files_directory = "./resumes"
   persist_directory = "./vector_db"
   collection_name = "resumes_collection"

   # Initialize the vector database handler
   vector_db_handler = VectorDBHandler(files_directory, persist_directory, collection_name)
   
   # Load or create the vector store database
   vector_db_handler.load_or_create_db()

   # Define the query for the search
   query = "I am looking for a software engineer with OpenAI hard skill."
   docs = vector_db_handler.query_vector_store(query)

   # Output the top result
   if docs:
       print("Top matching document:")
       print(docs[0].page_content)
   else:
       print("No matching documents found.")
  ```

