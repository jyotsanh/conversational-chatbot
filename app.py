import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import fitz
from langchain.schema import Document


# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
groq_api_key = os.getenv('GROQ_API_KEY')
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
def extract_pdf_text(uploaded_file):
    """
    Extract text from an uploaded PDF file using PyMuPDF (fitz).
    
    Args:
        uploaded_file (UploadedFile): Uploaded PDF file from Streamlit
    
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Read the uploaded file into a bytes buffer
        pdf_bytes = uploaded_file.getvalue()
        
        # Open the PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Initialize an empty string to store extracted text
        full_text = ""
        
        # Iterate through each page and extract text
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text from the page
            page_text = page.get_text()
            
            # Append page text to full text with a page break indicator
            full_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
        
        # Close the PDF document
        pdf_document.close()
        
        return full_text.strip()
    
    except Exception as e:
        # Handle potential errors during PDF extraction
        error_message = f"Error extracting text from PDF: {str(e)}"
        return error_message

# Sidebar setup
with st.sidebar:
    st.title('📄 PDF Chat Assistant')
    st.markdown('''
    ### Features
    - 📤 PDF Upload & Analysis
    - 💬 Intelligent Q&A
    - 🔍 Context-Aware Responses
    ''')
    add_vertical_space(3)
    st.write("🚀 Backend Project")

# Main page setup
st.header("💬 Chat with Your PDF or Just Chat")

# File upload
uploaded_file = st.file_uploader("Upload a PDF to chat about", type="pdf")

def create_vector_store(pdf_text):
    """
    Create a vector store with dynamically sized chunks based on PDF content length.
    
    Args:
        pdf_text (str): Text content of the PDF
    
    Returns:
        Retriever: Vector store retriever with optimized chunk sizing
    """
    # Ensure pdf_text is a string
    if isinstance(pdf_text, list):
        pdf_text = " ".join(pdf_text)
    
    # Calculate dynamic chunk sizing based on content length
    total_length = len(pdf_text)
    
    # Define chunk sizing strategy
    if total_length < 1000:
        # Short document
        chunk_size = 250
        chunk_overlap = 50
    elif total_length < 5000:
        # Medium document
        chunk_size = 500
        chunk_overlap = 100
    elif total_length < 20000:
        # Long document
        chunk_size = 1000
        chunk_overlap = 200
    else:
        # Very long document
        chunk_size = 1500
        chunk_overlap = 300
    
    # Create text splitter with dynamic sizing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len  # Use string length as measure
    )
    
    # Create a Document object
    doc = Document(page_content=pdf_text, metadata={
        'total_length': total_length,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    })
    
    # Split documents
    split_docs = text_splitter.split_documents([doc])
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create and return retriever
    docsearch = Chroma.from_documents(split_docs, embedding=embeddings,persist_directory=None)  # Use in-memory mode)
    
    # Add logging or display chunk information (optional)
    st.sidebar.info(f"""
    PDF Analysis:
    - Total Length: {total_length} characters
    - Chunk Size: {chunk_size}
    - Chunk Overlap: {chunk_overlap}
    - Total Chunks: {len(split_docs)}
    """)
    
    return docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'lambda_mult': 0.25}
    )
def create_qa_chain(retriever,groq_api_key):
    
    llm=ChatGroq(
             model_name="Llama3-8b-8192")

    prompt = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable assistant answering questions accurately and concisely.
    Respond as directly and informatively as possible based only on the information provided below. 
    Avoid phrases like "based on the context" and respond as if you have the answer directly.
    <context>
    {context}
    <context>
    
    Question: {input}
    
    Answer:
    """
    )


    document_chain = create_stuff_documents_chain(
        llm, 
        prompt
    ) 
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )
    return retrieval_chain

# Initialize session state for chat history and file content
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = None

# Handle PDF upload
if uploaded_file is not None:
    # Simulate processing the PDF (extract text, etc.)
    pdf_content = extract_pdf_text(uploaded_file)  # Replace this with actual PDF processing logic
    
    st.session_state.pdf_context = pdf_content
    st.session_state.retriever = create_vector_store(pdf_content)
    st.session_state.messages.append({"role": "assistant", "content": f"The PDF '{uploaded_file.name}' has been uploaded. Ask me anything about it!"})
        
else:
    st.session_state.retriever = None
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! I'm ready to chat or help you explore a PDF if you upload one."}
        ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt) 
    
    # Get response from LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Determine response generation method based on PDF context
                if st.session_state.retriever:
                    # If PDF is uploaded, use retrieval chain
                    retrieval_chain = create_qa_chain(
                        st.session_state.retriever, 
                        groq_api_key
                    )
                    response = retrieval_chain.invoke({
                        "input": prompt,
                        "context": st.session_state.pdf_context
                    })['answer']
                else:
                    # If PDF is not uploaded, use LLM directly
                    # If no PDF, use a standard chat model (you might want to add this)
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="Llama3-8b-8192"
                    )
                    response = llm.invoke([
                        SystemMessage(content="You are a helpful assistant."),
                        HumanMessage(content=prompt)
                    ]).content
                # Display and store response
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
            except Exception as e:
                error_response = f"An error occurred: {str(e)}"
                st.markdown(error_response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_response
                })
            # else:
            #     # If PDF is not uploaded, use LLM directly
            
            # retrieval_chain = create_qa_chain(retriever,groq_api_key)
            # response = retrieval_chain.invoke([
            #     SystemMessage(content=system_message_content),
            #     HumanMessage(content=prompt)
            # ]).content
            # st.markdown(response)
