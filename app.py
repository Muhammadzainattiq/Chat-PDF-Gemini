import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS #deprecated
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

genai.configure(api_key= st.secrets["GOOGLE_API_KEY"])


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 400)
    chunks  = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model= 'models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local('faiss_index')



def get_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template= prompt_template,input_variables=["context", "question"] )
    chain = load_qa_chain(llm = model, chain_type= 'stuff', prompt = prompt)
    return chain


def generate_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Gemini Chat PDF",page_icon=":card_index_dividers:")
    created_style = """
    color: #888888; /* Light gray color */
    font-size: 99px; /* Increased font size */
"""

    
st.markdown("""
    <style>
        .header {
            font-size: 46px;
            color: #1E90FF; /* Header color */
            text-align: center;
            margin-bottom: 30px;
        }
        .footer-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 50px;
        }
        .footer-text {
            color: #888888;
            font-size: 22px;
        }
        .link-container {
            text-align: right;
        }
        .link {
            display: inline-block;
            margin: 0 10px;
            padding: 5px 7px; /* Reduced padding */
            background-color: #f4f4f4;
            color: #333;
            text-decoration: none;
            font-weight: bold;
            border-radius: 5px;
            font-size: 12px; /* Reduced font size */
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .link:hover {
            background-color: #0073b1; /* LinkedIn color */
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


    st.markdown("""
        <div class="footer-container">
            <div class="footer-text">➡️ created by Muhammad Zain Attiq</div>
            <div class="link-container">
                <a class="link" href="mailto:zainatteeq@gmail.com" target="_blank">Email</a>
                <a class="link" href="https://www.linkedin.com/in/muhammadzainattiq/" target="_blank">LinkedIn</a>
                <a class="link" href="https://github.com/Muhammadzainattiq" target="_blank">GitHub</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.header("Chat with PDF using Gemini 🤖📄")
    
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type="pdf")
        if st.button ("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_chunks(raw_text)
                    if text_chunks:
                      get_vector_store(text_chunks)
                      st.success("Done")
                      st.info("Now You can query your pdf...")
                else:
                    st.write("couldnt get the raw text from pdfs")
    question = st.text_input("Enter the prompt:", value="What is there in this pdf?")                
    if question:
        try:
            generate_response(question)
        except Exception as e:
            st.info(e)

if __name__ == "__main__":
    main()
