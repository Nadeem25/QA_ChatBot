import streamlit as st
import os
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ['LANGSMITH_API_KEY']=os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING']="true"
os.environ["LANGSMITH_PROJECT"]=os.getenv('LANGSMITH_PROJECT')

# Step 1: Create Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temprature, max_token):
    openai.api_key=api_key
    llm=ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

# Title of the App
st.title("Enchanced Q&A Chatbot With OpenAI")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Open AI API Key:", type="password")

#  Drop down the select various Open AI model
llm=st.sidebar.selectbox("Select an Open AI Model", ["gpt-4o","gpt-4-turbo", 'gpt-4'])

# Adjust response parameter
temprature = st.sidebar.slider("Temprature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input, api_key, llm, temprature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")



