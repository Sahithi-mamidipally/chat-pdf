import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Set the page configuration as the first command
st.set_page_config(page_title="Chat PDF", page_icon="💁")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up dark mode state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

def inject_css(dark_mode):
    placeholder_color = "#ffffff" if dark_mode else "#666666"
    css = f"""
    <style>
    /* General Styling */
    body {{
        background-color: {"#1e1e1e" if dark_mode else "#D3F3F6"};
        color: {"#e0e0e0" if dark_mode else "#333333"};
    }}
    .stApp {{
        background-color: {"#1e1e1e" if dark_mode else "#D3F3F6"};
        color: {"#e0e0e0" if dark_mode else "#333333"};
    }}

    /* Header */
    .header {{
        font-size: 24px;
        font-weight: bold;
        color: {"#ffffff" if dark_mode else "#000000"};
        margin-bottom: 10px;
    }}

    /* Sidebar styling */
    .css-1d391kg.e1fqkh3o3 {{
        background-color: {"#333333" if dark_mode else "#ffffff"};
        color: {"#ffffff" if dark_mode else "#000000"};
        border-right: 1px solid {"#555555" if dark_mode else "#cccccc"};
    }}

    /* Text Input */
    .stTextInput input {{
        background-color: {"#333333" if dark_mode else "#ffffff"};
        color: {"#ffffff" if dark_mode else "#333333"};
        border-radius: 5px;
        padding: 10px;
    }}
    .stTextInput input::placeholder {{
        color: {placeholder_color};
    }}

    /* Button */
    .stButton>button {{
        background-color: {"#555555" if dark_mode else "#D3F3F6"};
        color: {"#ffffff" if dark_mode else "#00000"};
        border-radius: 5px;
        padding: 8px 16px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    provided context, say "answer is not available in the context"; do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("Chat with PDF using Accessibility Bot")

    user_question = st.text_input("Ask a Question from the PDF Files", placeholder="Type your question here...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.button("Switch to Dark Mode" if not st.session_state.dark_mode else "Switch to Light Mode", on_click=toggle_dark_mode)
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
