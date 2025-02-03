import streamlit as st
import time
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CONVERSATIONS_FILE = "conversations.json"

# Load stored conversations
if os.path.exists(CONVERSATIONS_FILE):
    with open(CONVERSATIONS_FILE, "r") as f:
        stored_conversations = json.load(f)
else:
    stored_conversations = {}

st.title("RAG Application built on Gemini Model")

# Session state for conversation history and conversations
if "conversations" not in st.session_state:
    st.session_state.conversations = stored_conversations  # Load stored conversations
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar to select or create a conversation
st.sidebar.title("Conversations")
new_chat = st.sidebar.button("New Conversation")

if new_chat:
    if len(st.session_state.conversations) >= 10:
        oldest_chat = min(st.session_state.conversations.items(), key=lambda x: x[1].get("last_visited", float('inf')))[0]
        confirm_delete = st.sidebar.checkbox(f"The oldest unvisited chat '{oldest_chat}' will be deleted. Proceed?")
        if not confirm_delete:
            st.stop()
        del st.session_state.conversations[oldest_chat]  # Remove the oldest unvisited chat
    
    chat_id = f"chat_{int(time.time())}"  # Temporary ID
    st.session_state.conversations[chat_id] = {"history": [], "pdf": None, "last_visited": time.time(), "name": "Untitled Chat"}
    st.session_state.current_chat = chat_id
    
    # Save to file
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(st.session_state.conversations, f)

# List available conversations with edit and delete options
for chat_id in list(st.session_state.conversations.keys()):
    chat_name = st.session_state.conversations[chat_id].get("name", chat_id)
    col1, col2, col3 = st.sidebar.columns([6, 2, 2])
    if col1.button(chat_name):
        st.session_state.current_chat = chat_id
        st.session_state.conversations[chat_id]["last_visited"] = time.time()
        with open(CONVERSATIONS_FILE, "w") as f:
            json.dump(st.session_state.conversations, f)
    if col2.button("✏️", key=f"edit_{chat_id}"):
        new_name = st.text_input("Rename chat:", value=chat_name, key=f"rename_{chat_id}")
        if st.button("Save", key=f"save_{chat_id}"):
            st.session_state.conversations[chat_id]["name"] = new_name
            with open(CONVERSATIONS_FILE, "w") as f:
                json.dump(st.session_state.conversations, f)
    if col3.button("🗑️", key=f"delete_{chat_id}"):
        del st.session_state.conversations[chat_id]
        with open(CONVERSATIONS_FILE, "w") as f:
            json.dump(st.session_state.conversations, f)
        st.experimental_rerun()

if not st.session_state.current_chat:
    st.write("Please create or select a conversation from the sidebar.")
    st.stop()

# PDF Upload Section for the current chat
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    chat_data = st.session_state.conversations[st.session_state.current_chat]
    chat_data["pdf"] = uploaded_file.name
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())
    
    loader = PyPDFLoader(uploaded_file.name)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    st.session_state.vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    
    # Save to file
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(st.session_state.conversations, f)

retriever = None
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Display chat history
st.write("### Chat History")
chat_data = st.session_state.conversations[st.session_state.current_chat]
for chat in chat_data["history"]:
    st.write(chat)

query = st.chat_input("Say something: ")

if query and retriever:
    if chat_data.get("name") == "Untitled Chat":
        chat_data["name"] = query[:30]  # First prompt becomes chat name
        with open(CONVERSATIONS_FILE, "w") as f:
            json.dump(st.session_state.conversations, f)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. If the Question is unrelated to the PDF, say that the question has no relation to the uploaded PDF."
        "Do not answer the question if it is unrelated to the PDF or if it is inappropriate. "
        "Use Five sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    history_text = "\n".join([msg for msg in chat_data["history"]][-10:])
    response = rag_chain.invoke({"input": f"{history_text}\n{query}"})
    answer = response["answer"]
    
    chat_data["history"].append(f"**You:** {query}")
    chat_data["history"].append(f"**AI:** {answer}")
    
    if len(chat_data["history"]) > 20:
        chat_data["history"].pop(0)
        chat_data["history"].pop(0)
    
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(st.session_state.conversations, f)
    
    st.write(f"**You:** {query}")
    st.write(f"**AI:** {answer}")
