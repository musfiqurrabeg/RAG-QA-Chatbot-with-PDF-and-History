## RAG Q&A Conversation With PDF Including Chat History


# ----------------------------- IMPORTS ---------------------------
# For building the interactive web app
import streamlit as st


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Vectorstore (Chroma DB) to store embeddings
from langchain_chroma import Chroma

# Keeps track of chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# For designing prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM integration with Groq API
from langchain_groq import ChatGroq

# Attach chain with chat history
from langchain_core.runnables.history import RunnableWithMessageHistory

# For Embedding Model
from langchain_huggingface import HuggingFaceEmbeddings

# To split documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# To load PDF documents
from langchain_community.document_loaders import PyPDFLoader
import os

# ----------------------------- ENVIRONMENT SETUP -----------------------------
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file (like HF_TOKEN)

# Set HuggingFace token from environment
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Initialize embeddings using HuggingFace model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
# ----------------------------- STREAMLIT PAGE SETUP -----------------------------
st.title("RAG Q&A with Chat History")
st.write("Upload your PDF and ask questions about its content. The system will remember the context of your previous questions.")

# Input field for Groq API key (required for ChatGroq LLM)
# api_key = st.text_input('Enter your groq api key: ', type="password")
api_key = os.getenv("GROQ_API_KEY")

## check groq api key is provided
if api_key:
    # Initialize Groq LLM with given API key and chosen model
    llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")

    # Session ID for tracking chat history across conversations
    session_id = st.text_input('Enter your session id: ', value='default_session')
    
    if 'store' not in st.session_state: # Store chat histories across multiple sessions
        st.session_state.store = {}

    # File uploader to allow user to upload multiple PDF files
    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)
    

    # ----------------------------- PROCESS UPLOADED PDFS -----------------------------
    if uploaded_files:
        documents = [] # List to hold all loaded documents

        # Iterate over each uploaded PDF file
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name # capture original File Name

            # Load the PDF file using PyPDFLoader
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs) # Add to overall document list
        
        # ----------------------------- CHUNKING & VECTORSTORE -----------------------------
        # Split documents into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Store embeddings inside Chroma vectorstore for semantic search
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

        retriever = vectorstore.as_retriever() # Retriever to fetch relevant context chunks based on query
        
        # ----------------------------- CONTEXTUALIZED QUESTION REFORMULATION -----------------------------
        # NOW ALL THE RETRIEVER-DEPENDENT CODE GOES HERE (PROPERLY INDENTED)
        contextualize_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        # Build contextualization prompt template
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"), # Insert chat history Dynamically
                ("human", "{input}"),
            ]
        )
        
        # Create a retriever that reformulates questions based on chat history
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        # ----------------------------- QA CHAIN SETUP -----------------------------
        # System prompt instructing LLM how to answer
        system_prompt = ( 
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        # Question-answering prompt template
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Create a chain that stuffs retrieved documents into the prompt
        ques_ans_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt
        )

        # Full retrieval chain: combines retriever + document answer chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=ques_ans_chain
        )


        # ----------------------------- SESSION HISTORY HANDLER -----------------------------
        # Function to return session-specific chat history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        

        # Attach retrieval chain with message history (so it remembers conversations)
        conversional_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # ----------------------------- USER QUESTION HANDLING -----------------------------
        # Input box for user queries
        user_question = st.text_input("Ask questions about your PDF file:")
        if user_question:

            session_history = get_session_history(session_id) # Get current session history
            
            # Invoke the chain with user query and session ID for history tracking
            response = conversional_rag_chain.invoke(
                {"input": user_question},
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            st.write("Assistant: ", response['answer'])             # Display assistant answer
            st.write("Chat History: ", session_history.messages)    # Show the stored chat history (all previous Q&A)
    
    else:
        st.info("Please upload a PDF file to start asking questions.")
# ----------------------------- ERROR HANDLING -----------------------------
else:
    st.warning("Please enter your Groq API key to proceed.")