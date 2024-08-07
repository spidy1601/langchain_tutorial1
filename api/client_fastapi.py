import requests
import streamlit as st
import time

# Function to get response from the CourseGPT API
def get_CourseGPT_response(input_text, session_id):
    response = requests.post(
        "http://10.144.113.132:7000/query",
        json={'prompt': input_text, 'session_id': session_id}
    )
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()['response']

# Streamlit framework
st.title('Course GPT')

# Initialize session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = "session_" + str(int(time.time()))

# Input text box for user queries
input_text = st.text_input("Ask a question related to the course")

# Display response if there's an input
if input_text:
    try:
        response = get_CourseGPT_response(input_text, st.session_state.session_id)
        st.write(response)
    except requests.RequestException as e:
        st.error(f"Error: {e}")
