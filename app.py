import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()

## Langchain Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING"] = os.getenv("LANGCHAIN_TRACING")
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with LLM"
api_key = os.getenv("GOOGLE_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are  helpful assistant. Please respond to user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    # openai.api_key=api_key
    # llm = ChatOpenAI(model=llm)
    llm = ChatGoogleGenerativeAI(model=llm, google_api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})

    return answer

## Title
st.title("Enhanced Q&A Chatbot With Gemini")

## Sidebar for settings
st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your Gemini API Key: ", type="password")

## Dropdown 
llm = st.sidebar.selectbox("Select an Gemini Model", ["gemini-1.5-pro", "gemini-1.5-flash"])

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide query")