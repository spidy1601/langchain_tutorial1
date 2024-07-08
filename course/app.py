import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import time
import pickle

load_dotenv()

##load the Groq API key
groq_api_key=os.environ["GROQ2_API_KEY"]

os.environ["LANGCHAIN_TRACING_V2"]="true"

LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT="Conestoga_FAQ"


with open('transcriptions_embeddings_json.pkl', 'rb') as f:
    loaded_embeddings = pickle.load(f)

st.title("Course Transcription AI ChatBot")
llm = ChatGroq(groq_api_key=groq_api_key,model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}

    """
)

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = loaded_embeddings.as_retriever(search_kwargs={"k": 10})
retriever_chain = create_retrieval_chain(retriever,document_chain)

prompt =st.text_input("Input your prompt here")

if prompt:
    start= time.process_time()
    response = retriever_chain.invoke({"input":prompt})
    print("Response Time: ",time.process_time()-start)
    st.write(response['answer'])

    #With a streamlit expander
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.metadata)
            st.write("---------------------------------------------------------")


# What can I bring to the Orientation?

# Can I use my ONE Card as a transit pass?

# When should I arrive for Convocation Ceremony?

# What if I can't attend my mandatory Online Academic Orientation Session? Will the session be recorded?

# I am an International STudent, Can I use One Card for transit?

# When is my Ceremony?