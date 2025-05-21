import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

# Load retriever
vectorstore = FAISS.load_local(
    "chatbot/faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define sample questions
questions = [
    "Who wrote The Hobbit?",
    "What is the average rating of The Catcher in the Rye?",
    "Tell me about a book by J.K. Rowling.",
    "Which book has the highest rating?",
    "Give me a fantasy book recommendation.",
]

# Collect answers
qa_pairs = []
for q in questions:
    answer = qa.run(q)
    qa_pairs.append((q, answer))

# Save to Excel
df = pd.DataFrame(qa_pairs, columns=["Question", "Answer"])
df.to_excel("chatbot/qa_samples.xlsx", index=False)

print("✅ Sample Q&A saved to chatbot/qa_samples.xlsx")
