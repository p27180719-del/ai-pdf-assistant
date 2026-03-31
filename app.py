import streamlit as st
import PyPDF2
import os

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# 🔑 Get API key from Streamlit Cloud
api_key = os.getenv("AIzaSyBcW484FjM-UEunUf-aaCvFI-qc4f2T64c")

st.set_page_config(page_title="AI PDF Assistant", layout="wide")

st.title("📄 AI PDF Assistant")
st.write("Upload a PDF and ask questions")


if "history" not in st.session_state:
    st.session_state.history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text


def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    return docs


def create_qa_chain(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    prompt_template = """
    You are a helpful AI assistant.
    Answer ONLY from the given context.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Give a short clear answer.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = read_pdf(uploaded_file)
    docs = split_text(text)
    st.session_state.qa_chain = create_qa_chain(docs)
    st.success("✅ PDF processed!")


query = st.chat_input("Ask something")

if query:
    st.session_state.history.append(("user", query))

    if st.session_state.qa_chain is None:
        answer = "Upload PDF first"
    else:
        response = st.session_state.qa_chain.invoke({"query": query})
        answer = response["result"]

    st.session_state.history.append(("assistant", answer))


for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)