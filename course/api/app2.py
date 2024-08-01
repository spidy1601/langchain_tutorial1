from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import time

load_dotenv()

## Load the Groq API key
groq_api_key = os.environ["GROQ2_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "Conestoga_FAQ"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db3 = Chroma(persist_directory="../chroma_db copy", embedding_function=embeddings)

metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="The topic of that talk which is captured in a transcription form. One of ['Unit 10 - Topic 1.webm','Unit 10 - Topic 2.webm','Unit 10 - Topic 3.webm','Unit 2 - Intro & Getting Started with OOL.webm','Unit 2.1 - Topic 1 - Defining Projects.webm','Unit 2.1 - Topic 2 - The Purpose of Projects.webm','Unit 2.1 - Topic 3 - Benefits of Project Management.webm','Unit 2.1 - Topic 4 - Project Lifecycles.webm','Unit 2.1 - Topic 5 - Project Processes & Knowledge.webm','Unit 2.1 - Topic 6 - Organizational Project Management.webm','Unit 2.1 - Topic 7 - The Triple Constraint.webm','Unit 2.2 - Ethics & Professional Conduct.webm','Unit 2.2 - The PMI Values.webm','Unit 2.2 - Topic 1 - Ethics & Professional Conduct.webm','Unit 2.2 - Topic 2 - The PMI Values.webm','Unit 2.3 - Topic 1 - Project Management Roles.webm','Unit 2.3 - Topic 2 - PM as Conductor.webm','Unit 2.3 - Topic 3 - PM Spheres of Influence.webm','Unit 2.3 - Topic 4 - PM Competencies.webm','Unit 3.1 - Topic 1 - Project Integration Management.webm','Unit 3.1 - Topic 2 - Develop Project Charter.webm','Unit 3.1 - Topic 3 - Develop PM Plan.webm','Unit 3.1 - Topic 4 - Direct and Manage Project Work.webm','Unit 3.1 - Topic 5 - Manage Project Knowledge.webm','Unit 3.1 - Topic 6 - Monitor and Control Project Work.webm','Unit 3.1 - Topic 7 - Perform Integration Change Control.webm','Unit 3.1 - Topic 8 - Closing Project or Phase.webm','Unit 3.2 - Topic 1 - Scope Management.webm','Unit 3.2 - Topic 2 - Schedule Managment.webm','Unit 3.2 - Topic 3 - Cost Management.webm','Unit 4 - Topic 1 - Project Quality Management.webm','Unit 4 - Topic 2 - Manage Quality.webm','Unit 4 - Topic 3 - Project Resource Management.webm','Unit 4 - Topic 4 - Manage Team.webm','Unit 4 - Topic 5 - Project Risk Management.webm','Unit 4 - Topic 6 - Identify Risks.webm','Unit 4 - Topic 7 - Project Communication Management.webm','Unit 4 - Topic 8 - Manage Communications.webm','Unit 4 - Topic 9 - Project Stakeholder Management.webm','Unit 5 - Topic 2 - Ethics.webm','Unit 5 - Topic 3 - Guiding Principles.webm','Unit 5- Topic 1 - Understanding SM.webm','Unit 6 - Topic 1 - SM Roles.webm','Unit 6 - Topic 2 - SM Definitions.webm','Unit 6 - Topic 3 - Service Relationship.webm','Unit 7 - 4 Dimensions of SM - Topic 3.webm','Unit 7 - 4 Dimesions of SM - Topic 1.webm','Unit 7 - 4 Dimesions of SM - Topic 4.webm','Unit 8 - Service Value - Topic 1.webm','Unit 8 - Service Value - Topic 2.webm','Unit 9 - Intro.webm','Unit 9 - Topic 1.webm','Unit 9 - Topic 2 - Part 1.webm','Unit 9 - Topic 2 - Part 2.webm','Unit 9 - Topic 3.webm','Unit1.webm','Week 1 Survey Results.webm']",
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="This is the file name from which have extracted transcription text from.",
        type="string",
    ),
]
document_content_description = "contains transcription of the video topics of Information Technology Operation, MGMT8680 - Spring 2020 course."

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

retriever = SelfQueryRetriever.from_llm(
    llm,
    db3,
    document_content_description,
    metadata_field_info,
    search_kwargs={"k": 15},
)

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API Server for Course Chat"
)

# List of allowed origins (i.e., frontend URLs that can make requests to this backend)
origins = [

    "http://localhost:4200", 
    "http://10.144.114.239:7000",

]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins to make requests
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    prompt: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    source: list

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response = conversational_rag_chain.invoke(
            {"input": request.prompt},
            config={"configurable": {"session_id": request.session_id}}
        )

        topic_set=set()
        final_sources=[]
        for con in response["context"]:
            topic_set.add(con.metadata["topic"])
        if len(topic_set)<=3:
            for topic in topic_set:
                final_sources.append(topic)
        else:
            final_sources = list(topic_set[:3])

        return QueryResponse(answer=response["answer"],source=final_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.192", port=7000)
