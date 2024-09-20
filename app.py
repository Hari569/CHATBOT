import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit app title
st.title("Generative AI App Using Langchain")

# Scraping and Document Loader
st.write("Loading data from website...")
loader = WebBaseLoader("https://englishfirm.com/")
docs = loader.load()

# Split documents into chunks
st.write("Splitting documents into chunks...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Embeddings
st.write("Creating embeddings...")
hf_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store
st.write("Building vector store...")
vectorstoredb = FAISS.from_documents(documents, embeddings)

# Query input
query = st.text_input("Enter your query", value="What is PTE?")

if query:
    # Retriever
    retriever = vectorstoredb.as_retriever()

    # LLM setup
    st.write("Setting up the LLM and retrieval chain...")
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        """
    )

    # Document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get response
    response = retrieval_chain.invoke({"input": query})
    
    st.write("### Answer:")
    st.write(response['answer'])

    st.write("### Context:")
    st.write(response['context'])
