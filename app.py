import os
import faiss
import numpy as np
import streamlit as st
import requests
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew
import ollama
from dotenv import load_dotenv

# Set up Streamlit UI
st.title("Agentic RAG using Crew AI")

# Initialize Sentence Transformers model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Firecrawl API configuration
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v0/search"

MODEL_NAME = os.getenv("MODEL_NAME", "ollama/llama3")  # Default to "ollama/llama3" if not set

# Load and process documents (support both PDF and TXT)
def load_and_process_documents(file_path):
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

# Create FAISS vector store with IndexIVFFlat for faster search
def create_faiss_index(texts):
    if not texts:
        raise ValueError("No text chunks found in the document.")
    embeddings = embedding_model.encode([text.page_content for text in texts])
    dimension = len(embeddings[0])
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(embeddings)))
    index.train(np.array(embeddings))
    index.add(np.array(embeddings))
    return index, texts

# Search in FAISS index with relevance threshold
def search_faiss_index(query, index, texts, k=3, threshold=0.7):
    query_embedding = embedding_model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), k)
    
    relevant_results = []
    for i, distance in zip(indices[0], distances[0]):
        if distance < threshold:  # Lower distance means higher relevance
            relevant_results.append(texts[i].page_content)
    
    return relevant_results

# Search online using Firecrawl API
def search_online(query):
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "max_results": 1
    }
    try:
        response = requests.post(FIRECRAWL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        if results and "results" in results:
            return results["results"][0]["content"]
        else:
            return "No results found online."
    except Exception as e:
        return f"Error searching online: {str(e)}"

# Define CrewAI agents
def create_crewai_agents():
    retriever_agent = Agent(
        role="Document Retriever",
        goal="Retrieve relevant information from the document database",
        backstory="An AI agent specialized in retrieving information from documents.",
        verbose=True,
        llm=MODEL_NAME
    )

    online_searcher_agent = Agent(
        role="Online Searcher",
        goal="Search the web for information if it is not found in the documents",
        backstory="An AI agent specialized in searching the web for information.",
        verbose=True,
        llm=MODEL_NAME
    )

    return retriever_agent, online_searcher_agent

# Generate response using Llama3 (Ollama)
def generate_response_with_llama3(query, context):
    prompt = f"Query: {query}\n\nContext: {context}\n\nAnswer:"
    response = ollama.generate(model="llama3", prompt=prompt)
    return response["response"]

# Main function to handle queries
def handle_query(query, index, texts):
    retriever_agent, online_searcher_agent = create_crewai_agents()

    # Task 1: Retrieve information from documents
    retrieve_task = Task(
        description=f"Retrieve relevant information for the query: {query}",
        agent=retriever_agent,
        expected_output="A list of relevant document chunks or a message indicating no relevant information was found."
    )

    # Task 2: Search online if no relevant information is found
    search_task = Task(
        description=f"Search online for the query: {query}",
        agent=online_searcher_agent,
        expected_output="The content of the most relevant online result or a message indicating no results were found."
    )

    # Step 1: Execute document retrieval task first
    retriever_crew = Crew(
        agents=[retriever_agent],
        tasks=[retrieve_task],
        verbose=True
    )
    retriever_result = retriever_crew.kickoff()

    # Check if retrieval found relevant information
    relevant_chunks = search_faiss_index(query, index, texts)
    if relevant_chunks:  # If relevant information is found in the document
        context = "\n".join(relevant_chunks)
        source = "Retrieved from document"
    else:  # If no relevant information is found, perform online search
        online_crew = Crew(
            agents=[online_searcher_agent],
            tasks=[search_task],
            verbose=True
        )
        online_result = online_crew.kickoff()
        context = online_result
        source = "Retrieved from online search"

    # Generate response using Llama3
    response = generate_response_with_llama3(query, context)
    # Append source information to the response
    full_response = f"{response}\n\n**Source:** {source}"
    return full_response

# Streamlit app
def main():
    with st.sidebar:
        st.header("Document Input")
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
        if uploaded_file is not None:
            file_path = os.path.join("/tmp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                texts = load_and_process_documents(file_path)
                index, texts = create_faiss_index(texts)
                st.success("Document processed and FAISS index created!")
                
                st.subheader("Document Preview")
                with st.expander("View Document Content"):
                    for i, text in enumerate(texts):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(text.page_content)
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    st.header("Chat with Agentic RAG")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if uploaded_file is not None:
        query = st.chat_input("Enter your query:")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            response = handle_query(query, index, texts)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
    else:
        st.warning("Please upload a .txt or .pdf document to start chatting.")

if __name__ == "__main__":
    main()