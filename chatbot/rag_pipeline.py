import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Load the books.csv file
df = pd.read_csv("data/books.csv")

# 2. Create simple knowledge base chunks from title, author, and rating
documents = []
for _, row in df.iterrows():
    text = f"Title: {row['title']}\nAuthor: {row['authors']}\nAvg Rating: {row['average_rating']}"
    documents.append(Document(page_content=text))

# 3. Split text into chunks if needed
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = splitter.split_documents(documents)

# 4. Embed and store in FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 5. Save the retriever to disk
vectorstore.save_local("chatbot/faiss_index")

print("✅ Vectorstore built and saved successfully!")
