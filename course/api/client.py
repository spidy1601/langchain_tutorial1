import requests
import streamlit as st

# def get_openai_response(input_text):
#     response=requests.post("http://localhost:8000/essay/invoke",
#     json={'input':{'topic':input_text}})

#     return response.json()['output']['content']

def get_CourseGPT_response(input_text):
    response=requests.post(
    "http://10.144.122.125:7000/chat/invoke",
    json={'input':input_text})

    return response.json()['output']

## streamlit framework

st.title('Course GPT')
#input_text1=st.text_input("Write an essay on")
input_text=st.text_input("Ask question related to the course")

# if input_text1:
#     st.write(get_openai_response(input_text))

if input_text:
    st.write(get_CourseGPT_response(input_text))