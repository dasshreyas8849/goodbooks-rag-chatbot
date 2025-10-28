# chatbot/streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment
load_dotenv()

# Load vectorstore
vectorstore = FAISS.load_local(
    "chatbot/faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

# Setup retriever + model
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="📚 BookBot", page_icon="📖")
st.title("📚 BookBot")
st.write("Ask anything about books in the Goodbooks-10k dataset!")

user_question = st.text_input("🔎 Ask a question:")

if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(user_question)
    st.markdown("### 🤖 Answer:")
    st.write(response)
