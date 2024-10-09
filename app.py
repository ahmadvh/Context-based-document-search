from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate     
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import pandas as pd

# Load environment variables
load_dotenv()

# Define a directory of .txt files that we want to perform a context-based search on
files_directory = "./resumes"

# Create an empty DataFrame with columns for file path and content
df = pd.DataFrame(columns=["file_path", "content"])

# Iterate over all .txt files in the directory
for file_name in os.listdir(files_directory):
    if file_name.endswith(".txt"):
        file_path = os.path.join(files_directory, file_name)
        
        # Read the content of the file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Add the file path and content to the DataFrame
        df = df._append({"file_path": file_path, "content": content}, ignore_index=True)

# define the embedding model and generating the vector db.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=512,
        )

vectors = embedding_model.embed_documents(df.content.to_list())
df['embeddings'] = vectors
df.to_csv("knowledge_base.csv", index = False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~