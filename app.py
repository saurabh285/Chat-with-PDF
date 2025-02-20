import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
from io import BytesIO
 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
 
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
# Extracting all the text from the PDFs and storing it in text
def get_pdf_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        try:
            pdf_bytes = uploaded_file.read()  # Read file into bytes
            pdf_file = io.BytesIO(pdf_bytes)  # Create a file-like object from bytes
            pdf_reader = PdfReader(pdf_file)  # Read the PDF from the file-like object
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
    return text
 
# Making the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
 
# Storing the text in vector database form in our local storage
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
 
# The language model (LLM) takes the documents retrieved by the similarity search (stored in the docs variable) and uses them as context to generate a detailed and coherent answer to the user's question.
def get_conversational_chain():
    prompt_template = """
    Answer the Question as detailed as possible from the provided context
    Context: \n{context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
 
# When the chain is returned from get_conversational_chain, it is an object that can take inputs in the form of a dictionary. This dictionary should contain the necessary inputs (such as the input_documents and question in this case) required to generate the response.
# The chain object returned by get_conversational_chain is already predefined and configured with the necessary prompt template, model, and chain type. Once this chain object is returned, you can pass the required parameters to it to generate a response.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"]) # Print the output text from the response list
 
# Streamlit setup
def main():
    st.set_page_config(page_title="Chat with Multiple PDF", layout="wide")
    st.header("Chat with Multiple PDFs using Gemini")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")
 
if __name__ == "__main__":
    main()