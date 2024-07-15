import requests
import streamlit as st

# def get_openai_response(input_text):
#     response=requests.post("http://localhost:8000/essay/invoke",
#     json={'input':{'topic':input_text}})

#     return response.json()['output']['content']

def get_ollama_response(input_text):
    response=requests.post(
    "http://localhost:8000/poem/invoke",
    json={'input':{'question':input_text}})

    return response.json()

## streamlit framework

st.title('Course GPT')
#input_text1=st.text_input("Write an essay on")
input_text=st.text_input("Ask question related to the course")

# if input_text1:
#     st.write(get_openai_response(input_text))

if input_text:
    st.write(get_ollama_response(input_text))