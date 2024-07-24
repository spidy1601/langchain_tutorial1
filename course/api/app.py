from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os

from langchain_community.llms import Ollama

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser


import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chains.retrieval import create_retrieval_chain
from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain_community import query
import time
import pickle

load_dotenv()

#os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

os.environ['GROQ_API_KEY']=os.getenv("GROQ2_API_KEY")

groq_api_key=os.environ["GROQ2_API_KEY"]

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API Server for Course Chat"
)

##load the Groq API key




embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db3 = Chroma(persist_directory="../chroma_db copy",embedding_function=embeddings)

# with open("D:/VARLAB/langchain_tutorial1/course/transcriptions_embeddings_json.pkl","rb") as f:
#     db3=pickle.load(f)


metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="The Information Technology Operation, MGMT8680 - Spring 2020 course which is captured in a transcription form has many topics and have Units from this list - ['Unit 10 - Topic 1.webm','Unit 10 - Topic 2.webm','Unit 10 - Topic 3.webm','Unit 2 - Intro & Getting Started with OOL.webm','Unit 2.1 - Topic 1 - Defining Projects.webm','Unit 2.1 - Topic 2 - The Purpose of Projects.webm','Unit 2.1 - Topic 3 - Benefits of Project Management.webm','Unit 2.1 - Topic 4 - Project Lifecycles.webm','Unit 2.1 - Topic 5 - Project Processes & Knowledge.webm','Unit 2.1 - Topic 6 - Organizational Project Management.webm','Unit 2.1 - Topic 7 - The Triple Constraint.webm','Unit 2.2 - Ethics & Professional Conduct.webm','Unit 2.2 - The PMI Values.webm','Unit 2.2 - Topic 1 - Ethics & Professional Conduct.webm','Unit 2.2 - Topic 2 - The PMI Values.webm','Unit 2.3 - Topic 1 - Project Management Roles.webm','Unit 2.3 - Topic 2 - PM as Conductor.webm','Unit 2.3 - Topic 3 - PM Spheres of Influence.webm','Unit 2.3 - Topic 4 - PM Competencies.webm','Unit 3.1 - Topic 1 - Project Integration Management.webm','Unit 3.1 - Topic 2 - Develop Project Charter.webm','Unit 3.1 - Topic 3 - Develop PM Plan.webm','Unit 3.1 - Topic 4 - Direct and Manage Project Work.webm','Unit 3.1 - Topic 5 - Manage Project Knowledge.webm','Unit 3.1 - Topic 6 - Monitor and Control Project Work.webm','Unit 3.1 - Topic 7 - Perform Integration Change Control.webm','Unit 3.1 - Topic 8 - Closing Project or Phase.webm','Unit 3.2 - Topic 1 - Scope Management.webm','Unit 3.2 - Topic 2 - Schedule Managment.webm','Unit 3.2 - Topic 3 - Cost Management.webm','Unit 4 - Topic 1 - Project Quality Management.webm','Unit 4 - Topic 2 - Manage Quality.webm','Unit 4 - Topic 3 - Project Resource Management.webm','Unit 4 - Topic 4 - Manage Team.webm','Unit 4 - Topic 5 - Project Risk Management.webm','Unit 4 - Topic 6 - Identify Risks.webm','Unit 4 - Topic 7 - Project Communication Management.webm','Unit 4 - Topic 8 - Manage Communications.webm','Unit 4 - Topic 9 - Project Stakeholder Management.webm','Unit 5 - Topic 2 - Ethics.webm','Unit 5 - Topic 3 - Guiding Principles.webm','Unit 5- Topic 1 - Understanding SM.webm','Unit 6 - Topic 1 - SM Roles.webm','Unit 6 - Topic 2 - SM Definitions.webm','Unit 6 - Topic 3 - Service Relationship.webm','Unit 7 - 4 Dimensions of SM - Topic 3.webm','Unit 7 - 4 Dimesions of SM - Topic 1.webm','Unit 7 - 4 Dimesions of SM - Topic 4.webm','Unit 8 - Service Value - Topic 1.webm','Unit 8 - Service Value - Topic 2.webm','Unit 9 - Intro.webm','Unit 9 - Topic 1.webm','Unit 9 - Topic 2 - Part 1.webm','Unit 9 - Topic 2 - Part 2.webm','Unit 9 - Topic 3.webm','Unit1.webm','Week 1 Survey Results.webm']",
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="This is the file name from which have extracted transcription text from.",
        type="string",
    ),
]
document_content_description = "contains transcription of the video topics of Information Technology Operation, MGMT8680 - Spring 2020 course taught by Instructor/Professor Sean Yo"

llm = ChatGroq(groq_api_key=groq_api_key,model_name="mixtral-8x7b-32768")

retriever = SelfQueryRetriever.from_llm(
    llm,
    db3,
    document_content_description,
    metadata_field_info,
    search_kwargs={"k": 3},
)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide a good response based on the question and context
    <context>
    {context}
    </context>
    Questions:{question}

    """
)

document_chain = create_stuff_documents_chain(llm,prompt)
# retriever = db3.as_retriever()
retriever_chain = create_retrieval_chain(retriever,document_chain)
entry_point_chain=RunnableParallel({"context":retriever,"question":RunnablePassthrough()})


add_routes(
    app,
    entry_point_chain|prompt|llm|StrOutputParser(),
    path="/chat"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=7000)