# General Purpose AI Assistant

An intelligent AI assistant built with LangGraph, LangChain, and Gradio that combines RAG (Retrieval-Augmented Generation), web search, and document creation capabilities.

## Features

### 🤖 **Three Powerful Tools:**

1. **📚 Document Retrieval (RAG)**
   - Upload PDFs to build a knowledge base
   - Semantic search through uploaded documents
   - Uses ChromaDB for vector storage
   - Powered by HuggingFace embeddings

2. **🌐 Web Search**
   - Real-time web search using Tavily API
   - Get current information and recent news
   - Comprehensive search results with sources

3. **📝 Document Creation**
   - Generate Word documents (.docx)
   - Download created documents directly
   - Formatted and professional output

## Architecture

### **File Structure:**
```
app/
├── tools.py       # Tool definitions (RAG, Search, Document Creation)
├── agent.py       # LangGraph workflow and agent logic
└── app.py         # Gradio web interface

requirements.txt   # Python dependencies
.env              # Environment variables (create this)
```

### **Technology Stack:**
- **LangGraph**: Agent workflow orchestration
- **LangChain**: Tool integration and RAG pipeline
- **Gradio**: Web interface
- **ChromaDB**: Vector database for document storage
- **Groq**: Fast LLM inference
- **Tavily**: Web search API

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your-groq-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

### 3. Get API Keys

- **Groq API Key**: [Groq Console](https://console.groq.com/keys)
- **Tavily API Key**: [Tavily](https://tavily.com/)

## Usage

### Start the Application

```bash
python app/app.py
```

The application will be available at `http://localhost:7860`

### Using the Assistant

1. **Chat Normally**: Ask any question, just like a regular chatbot
2. **Upload PDFs**: Add documents to expand the knowledge base
3. **Web Search**: Ask about current events or recent information
4. **Create Documents**: Request document creation and download them

### Example Interactions

**Document Retrieval:**
```
You: What does the uploaded document say about machine learning?
AI: [Searches uploaded PDFs and provides relevant information]
```

**Web Search:**
```
You: What's the latest news about AI?
AI: [Uses Tavily to search the web for current information]
```

**Document Creation:**
```
You: Create a summary document of our conversation
AI: [Generates a Word document that you can download]
```

## How It Works

### LangGraph Workflow

```
START → agent → conditional edge → [tools | END]
tools → agent
```

1. **Agent Node**: Processes user messages and decides which tool to use
2. **Tool Selection**: Chooses between retrieve_documents, web_search, or create_document
3. **Tool Execution**: Executes the selected tool
4. **Response Generation**: Returns results to the user

### RAG Pipeline

1. PDF uploaded → Text extracted using PyMuPDF4LLM
2. Text chunked → RecursiveCharacterTextSplitter (500 chars, 150 overlap)
3. Embeddings created → HuggingFace BGE-small-en-v1.5
4. Stored in ChromaDB → Persistent vector database
5. Query → Semantic search → Relevant chunks returned

### Tool Selection Logic

The agent automatically selects the appropriate tool based on:
- **Document-related questions** → retrieve_documents
- **Current events/web info** → web_search
- **Document creation requests** → create_document

## Project Structure

### **tools.py**
Contains three main tools as LangChain tools:
- `retrieve_documents(query, collection_name, top_k)`: RAG retrieval
- `web_search(query)`: Tavily web search
- `create_document(content, filename, title)`: Word document creation

### **agent.py**
LangGraph agent implementation:
- Agent state management
- Tool binding with LLM
- Workflow definition
- Conversation memory

### **app.py**
Gradio web interface:
- Chat interface
- PDF upload functionality
- Document download section
- State management

## Features in Detail

### Document Upload
- Supports multiple PDF uploads
- Automatic text extraction and chunking
- Persistent storage in ChromaDB
- Real-time status updates

### Chat Interface
- Streaming responses
- Message history
- Context-aware conversations
- Tool usage transparency

### Document Downloads
- Generated documents stored in `documents/` folder
- Download directly from interface
- Refresh to see new documents
- Professional Word format

## Configuration

### Adjustable Parameters

**In `tools.py`:**
- `top_k`: Number of RAG results (default: 5)
- `max_results`: Web search results (default: 5)
- `search_depth`: Tavily search depth (default: "advanced")

**In `agent.py`:**
- `model`: Groq model (default: "llama-3.1-8b-instant")
- `temperature`: LLM creativity (default: 0.7)

**In `app.py`:**
- `chunk_size`: Text chunking size (default: 500)
- `chunk_overlap`: Chunk overlap (default: 150)
- `collection_name`: ChromaDB collection name

## Troubleshooting

### Common Issues

1. **"No API key" error**: Make sure `.env` file exists with valid API keys
2. **ChromaDB errors**: Delete `chroma_db/` folder and re-upload documents
3. **Import errors**: Run `pip install -r requirements.txt`
4. **Document not found**: Check `documents/` folder exists

### Debug Mode

Set environment variable for verbose logging:
```bash
export LANGCHAIN_VERBOSE=true
python app/app.py
```

## License

MIT License - see LICENSE file for details.

## Contributing

Feel free to submit issues and enhancement requests!
