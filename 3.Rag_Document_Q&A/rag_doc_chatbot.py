import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

# Load the Groq API
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Step 1: Initialize Model
groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Step 2: Initialize Prompt
# prompt=ChatPromptTemplate.from_messages(
#     """ m
#     Answer the question based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Question:{input}
# """
# )

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """Answer the question based on the provided context only.
        Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>
    Question: {input}"""
    )
])

def create_vectore_embedding():
    if "vectores" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_paper") # Data Ingestion
        st.session_state.docs=st.session_state.loader.load() # Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Splits dcouments technique
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) # Splits the documents
        st.session_state.vectores=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) # Convert the document into the vectores and store into the DB

user_prompt=st.text_input("Enter you query from the research paper")
if st.button("Documents Embedding"):
    create_vectore_embedding()
    st.write("Vector Database is ready")

if user_prompt:
    document_chain=create_stuff_documents_chain(groq_llm, prompt=prompt)
    retriever=st.session_state.vectores.as_retriever() # retriver is  an interface which use to make query to specific db
    retrieval_chain=create_retrieval_chain(retriever, document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input': user_prompt})
    #print(f"Response Time: {time.process_time()} - {start}")

    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similiraty Search"): 
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------------")