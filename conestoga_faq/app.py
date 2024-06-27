import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
import pickle

load_dotenv()

##load the Groq API key
groq_api_key=os.environ["GROQ2_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model="llama3")
    st.session_state.loader = CSVLoader(file_path='./clean_FAQ.csv',encoding="utf8",
        csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['question', 'answer']
    })
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# loader=WebBaseLoader(["https://www.conestogac.on.ca/career-centre/faq","https://www.conestogac.on.ca/admissions/registrar-office/grading-transcripts","https://orientation.conestogac.on.ca/questions/faq","https://www.conestogac.on.ca/student-rights/faq","https://www.conestogac.on.ca/employment/applicant-faq","https://www.conestogac.on.ca/convocation/frequently-asked-questions","https://www.conestogac.on.ca/onecard/faq"])
# loader = CSVLoader(file_path=file_path)


st.title("Conestoga FAQ AI ChatBot")
llm = ChatGroq(groq_api_key=groq_api_key,model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    """
    Please provide a good response based on the question and context. Sometimes question is similar or same to the context's question. Answer according to the context only.
    <context>
    {context}
    </context>
    Questions:{input}

    """
)

# with open('embeddings.pickle', 'rb') as f:
#     loaded_embeddings = pickle.load(f)


document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 15})
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
            print(doc.page_content)
            print("---------------------------------------------------------")