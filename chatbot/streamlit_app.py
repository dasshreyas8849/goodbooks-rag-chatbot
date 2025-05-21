# chatbot/streamlit_app.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load vectorstore
vectorstore = FAISS.load_local(
    "chatbot/faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="📚 BookBot", page_icon="📖")
st.title("📚 BookBot")
st.write("Ask anything about books in the Goodbooks-10k dataset!")

# Input box
user_question = st.text_input("🔎 Ask a question:")

if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(user_question)
    st.markdown("### 🤖 Answer:")
    st.write(response)
