# chatbot/chatbot_app.py

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load API key from .env
load_dotenv()

# Load FAISS index
vectorstore = FAISS.load_local(
    "chatbot/faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

# Initialize the Chat Model
llm = ChatOpenAI(temperature=0)

# Set up Retrieval-QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask the user for a question
print("📚 Welcome to BookBot! Ask me anything about the books:")
while True:
    query = input("\nYour Question (or type 'exit'): ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break
    answer = qa_chain.run(query)
    print("🤖 Answer:", answer)
