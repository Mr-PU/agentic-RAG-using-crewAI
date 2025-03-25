### Project Description: Agentic RAG using CrewAI and Ollama

#### Overview
The "Agentic RAG using CrewAI" project is a Retrieval-Augmented Generation (RAG) system that leverages the CrewAI framework to orchestrate autonomous AI agents for answering user queries. It combines local document retrieval with online search capabilities, using a modular agent-based approach. The system processes uploaded documents (PDF or TXT), retrieves relevant information from them, and falls back to an online search if the information isn’t found locally. Responses are generated using a local language model (via Ollama), and users are informed whether the answer came from the document or the web.

#### Key Features
1. **Document Processing**: Accepts `.txt` and `.pdf` files, splits them into chunks, and indexes them using FAISS for efficient retrieval.
2. **Agentic Workflow**: Uses two CrewAI agents:
   - **Document Retriever**: Searches the uploaded document for relevant information.
   - **Online Searcher**: Performs a web search via the Firecrawl API if the document lacks relevant data.
3. **Conditional Search**: Online searches are triggered only if the document doesn’t contain relevant information.
4. **Source Attribution**: Informs users whether the response came from the document or an online search.
5. **Local LLM**: Uses Ollama to run a local language model (e.g., Llama3) for response generation.
6. **Streamlit UI**: Provides a user-friendly interface for uploading documents and interacting via a chat system.

#### Purpose
The project aims to demonstrate an agentic RAG system where AI agents collaborate to provide accurate, context-aware responses. It’s designed for users who want to query private documents locally while supplementing with online data when needed, all without relying on external API-based LLMs.

![Screenshot of Agentic RAG](https://github.com/Mr-PU/agentic-RAG-using-crewAI/blob/main/1.png?raw=true)


#### GitHub Repository
- **Link**: [https://github.com/Mr-PU/agentic-RAG-using-crewAI](https://github.com/Mr-PU/agentic-RAG-using-crewAI)
- **Assumption**: Since I can’t access the repository directly, I’ll base this description on the code you provided, assuming it matches the repository’s main script (e.g., `app.py`).

---

### Installation

#### Prerequisites
- **Operating System**: Linux or macOS (Ollama isn’t natively supported on Windows without WSL).
- **Python**: Version 3.8 or higher.
- **Hardware**: Sufficient RAM (8GB+ recommended) for running local LLMs via Ollama.

#### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mr-PU/agentic-RAG-using-crewAI.git
   cd agentic-RAG-using-crewAI
   ```

2. **Install Python Dependencies**:
   - Create a `requirements.txt` file with the following content (based on your code):
     ```
     faiss-cpu==1.8.0
     numpy==1.26.4
     streamlit==1.32.0
     requests==2.31.0
     langchain==0.1.16
     sentence-transformers==2.6.1
     crewai==0.28.8
     ollama==0.1.8
     pypdf==4.1.0
     python-dotenv==1.0.1
     ```
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the project root:
     ```bash
     touch .env
     ```
   - Add the following variables:
     ```
     FIRECRAWL_API_KEY=your-firecrawl-api-key
     MODEL_NAME=ollama/llama3
     ```
   - Replace `your-firecrawl-api-key` with your actual Firecrawl API key (get it from [firecrawl.dev](https://firecrawl.dev)).

4. **Install and Configure Ollama**:
   - **Install Ollama**:
     - On Linux/macOS, run:
       ```bash
       curl -fsSL https://ollama.com/install.sh | sh
       ```
     - This installs Ollama as a service.
   - **Pull the Llama3 Model**:
     - Download the `llama3` model (or another model specified in `.env`):
       ```bash
       ollama pull llama3
       ```
     - Verify it’s installed:
       ```bash
       ollama list
       ```
   - **Start Ollama Server**:
     - Run the server in a separate terminal:
       ```bash
       ollama serve
       ```
     - It runs on `localhost:11434` by default.

5. **Run the Application**:
   - Start the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Open your browser to `http://localhost:8501` to interact with the UI.

---

### Working of Ollama

#### What is Ollama?
Ollama is a platform for running large language models (LLMs) locally on your machine. It provides a simple interface to download, manage, and query models like Llama3, Mistral, etc., without requiring cloud-based APIs.

#### How Ollama Works in This Project
1. **Model Hosting**:
   - Ollama hosts the `llama3` model (or another specified in `.env`) locally. The `ollama pull llama3` command downloads it to your system.
2. **Server Operation**:
   - The `ollama serve` command starts a local server at `localhost:11434`, exposing an API for model inference.
3. **Python Integration**:
   - The `ollama` Python package (`import ollama`) acts as a client to communicate with this server. The `ollama.generate` function sends prompts to the model and retrieves responses.
4. **Role in the Project**:
   - Ollama powers the `generate_response_with_llama3` function, generating natural language answers based on retrieved context (from documents or online search).

#### Why Use Ollama?
- **Privacy**: Keeps data local, avoiding external API calls.
- **Cost**: Free to use with open-source models.
- **Customization**: Allows tweaking model parameters or using custom models.

---

### Detailed Explanation of How the Code Works

#### Code Structure
The script (`app.py`) is organized into several functions, each handling a specific part of the RAG pipeline. Here’s a detailed breakdown:

1. **Imports and Setup**:
   - Libraries like `faiss`, `numpy`, `streamlit`, `requests`, `langchain`, `sentence_transformers`, `crewai`, `ollama`, and `dotenv` are imported.
   - Environment variables are loaded with `load_dotenv()`.
   - The Sentence Transformer model (`all-MiniLM-L6-v2`) is initialized for embeddings.
   - Streamlit UI is set up with `st.title`.

   **Key Lines**:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
   MODEL_NAME = os.getenv("MODEL_NAME", "ollama/llama3")
   ```

2. **Document Processing**:
   - **`load_and_process_documents(file_path)`**:
     - Determines file type (`.txt` or `.pdf`) and uses `TextLoader` or `PyPDFLoader`.
     - Splits text into 500-character chunks with 100-character overlap using `RecursiveCharacterTextSplitter`.
     - Returns a list of document chunks.

   - **`create_faiss_index(texts)`**:
     - Encodes document chunks into embeddings using the Sentence Transformer.
     - Creates a FAISS `IndexIVFFlat` index with 100 clusters for fast similarity search.
     - Trains and adds embeddings to the index.

   **Key Logic**:
   ```python
   embeddings = embedding_model.encode([text.page_content for text in texts])
   index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(embeddings)))
   index.train(np.array(embeddings))
   index.add(np.array(embeddings))
   ```

3. **Retrieval Functions**:
   - **`search_faiss_index(query, index, texts, k=3, threshold=0.7)`**:
     - Encodes the query into an embedding.
     - Searches the FAISS index for the top 3 similar chunks.
     - Filters results with a distance threshold of 0.7 (lower is more relevant).
     - Returns relevant text chunks.

   - **`search_online(query)`**:
     - Makes a POST request to the Firecrawl API with the query.
     - Returns the content of the first result or an error message.

   **Key Logic**:
   ```python
   query_embedding = embedding_model.encode([query])[0]
   distances, indices = index.search(np.array([query_embedding]), k)
   if distance < threshold:
       relevant_results.append(texts[i].page_content)
   ```

4. **CrewAI Agents**:
   - **`create_crewai_agents()`**:
     - Defines two agents:
       - **Retriever Agent**: Searches the document database.
       - **Online Searcher Agent**: Searches the web if needed.
     - Both use the model specified in `MODEL_NAME` (from `.env`).

   **Key Lines**:
   ```python
   retriever_agent = Agent(role="Document Retriever", ..., llm=MODEL_NAME)
   online_searcher_agent = Agent(role="Online Searcher", ..., llm=MODEL_NAME)
   ```

5. **Response Generation**:
   - **`generate_response_with_llama3(query, context)`**:
     - Combines the query and context into a prompt.
     - Sends it to Ollama’s `llama3` model (or the specified model) for generation.
     - Returns the response.

   **Key Logic**:
   ```python
   prompt = f"Query: {query}\n\nContext: {context}\n\nAnswer:"
   response = ollama.generate(model="llama3", prompt=prompt)
   ```

6. **Query Handling**:
   - **`handle_query(query, index, texts)`**:
     - Creates agents and defines tasks for document retrieval and online search.
     - Executes the retrieval task first using `retriever_crew`.
     - Checks if `search_faiss_index` finds relevant chunks:
       - If yes, uses document chunks as context and sets source to "Retrieved from document".
       - If no, runs `online_crew` and sets source to "Retrieved from online search".
     - Generates a response with Llama3 and appends the source.

   **Key Logic**:
   ```python
   relevant_chunks = search_faiss_index(query, index, texts)
   if relevant_chunks:
       context = "\n".join(relevant_chunks)
       source = "Retrieved from document"
   else:
       online_result = online_crew.kickoff()
       context = online_result
       source = "Retrieved from online search"
   full_response = f"{response}\n\n**Source:** {source}"
   ```

7. **Streamlit UI**:
   - **`main()`**:
     - Sidebar: Uploads a `.txt` or `.pdf` file, processes it, and shows a preview.
     - Chat Interface: Displays chat history and accepts user queries.
     - On query submission, calls `handle_query` and updates the chat with the response.

   **Key Logic**:
   ```python
   if uploaded_file:
       texts = load_and_process_documents(file_path)
       index, texts = create_faiss_index(texts)
       query = st.chat_input("Enter your query:")
       if query:
           response = handle_query(query, index, texts)
           st.session_state.chat_history.append({"role": "assistant", "content": response})
   ```

#### Workflow
1. **User Uploads Document**: Uploaded file is saved to `/tmp`, processed into chunks, and indexed with FAISS.
2. **User Enters Query**: Query is sent to `handle_query`.
3. **Document Retrieval**: Retriever agent searches the FAISS index.
4. **Conditional Online Search**: If no relevant info is found, the online searcher queries Firecrawl.
5. **Response Generation**: Context (document or online) is fed to Llama3 via Ollama, and the response is returned with source info.
6. **UI Update**: Response is displayed in the chat with source attribution.

---

### Conclusion
This project showcases a robust agentic RAG system using CrewAI, FAISS, and Ollama. It balances local document retrieval with online search, powered by a local LLM for privacy and cost-efficiency. The Streamlit UI makes it accessible, while the source attribution enhances transparency. To extend it, you could add support for more file types, tune the FAISS threshold, or integrate additional tools for the agents.

Let me know if you need clarification or help with specific parts!
