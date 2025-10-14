"""
Gradio Interface for General Purpose AI Assistant

This module provides a web interface with:
- Chat interface for conversing with the AI agent
- PDF upload functionality for RAG
- Document download capabilities
"""

import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from agent import app as agent_app, get_system_message
from tools import process_and_store_pdf

# Load environment variables
load_dotenv()

# Global configuration
COLLECTION_NAME = "general_collection"
CHROMA_DIR = "./chroma_db"


def process_message(message, history):
    """Process user message through the agent."""
    try:
        # Create conversation config
        config = {"configurable": {"thread_id": "gradio_session"}}
        
        # Add system message to the first interaction
        if not history:
            system_msg = get_system_message()
            all_messages = [
                HumanMessage(content = system_msg),
                HumanMessage(content = message)
            ]
        else:
            all_messages = [HumanMessage(content = message)]
        
        # Run the agent
        response = agent_app.invoke(
            {"messages": all_messages},
            config = config
        )
        
        # Get the last AI message
        last_message = response["messages"][-1]
        
        if isinstance(last_message, AIMessage):
            return last_message.content
        
        return "I'm sorry, I couldn't process that request."
        
    except Exception as e:
        return f"Error: {str(e)}"


def upload_pdfs(files):
    """Handle PDF uploads and add to Chroma collection using LangChain utilities."""
    if not files:
        return "No files uploaded."
    
    try:
        processed_files = []
        total_chunks = 0
        
        for file in files:
            # Use the LangChain-based utility function from tools.py
            num_chunks = process_and_store_pdf(
                filepath = file.name,
                collection_name = COLLECTION_NAME,
                chunk_size = 500,
                chunk_overlap = 150
            )
            
            if num_chunks > 0:
                filename = os.path.basename(file.name)
                processed_files.append(filename)
                total_chunks += num_chunks
        
        if processed_files:
            file_list = ", ".join(processed_files)
            return f"✅ Successfully processed {len(processed_files)} file(s): {file_list}\n📦 Added {total_chunks} chunks to the knowledge base."
        else:
            return "❌ No files were successfully processed."
            
    except Exception as e:
        return f"❌ Error processing files: {str(e)}"


def list_documents():
    """List all available documents for download."""
    docs_dir = Path("documents")
    if not docs_dir.exists():
        return []
    
    # Get all .docx files and return as list of file paths
    doc_files = list(docs_dir.glob("*.docx"))
    return [str(f) for f in doc_files]


# Create Gradio interface
with gr.Blocks(title = "AI Assistant", theme = gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 General Purpose AI Assistant")
    gr.Markdown("""
    I'm an AI assistant with access to:
    - 📚 **Document Retrieval**: Search through uploaded PDFs
    - 🌐 **Web Search**: Find current information online
    - 📝 **Document Creation**: Generate downloadable Word documents
    
    Upload PDFs to expand my knowledge base, or just start chatting!
    """)
    
    with gr.Row():
        # Left Column - Chat
        with gr.Column(scale = 2):
            chatbot = gr.Chatbot(
                label = "Chat with the Assistant",
                height = 500,
                show_copy_button = True,
                type = 'messages'
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder = "Ask me anything...",
                    label = "Your Message",
                    scale = 4,
                    lines = 1
                )
                send_btn = gr.Button("Send", variant = "primary", scale = 1)
            
            # PDF Upload Section
            with gr.Accordion("📤 Upload PDFs", open = False):
                file_upload = gr.File(
                    label = "Upload PDF Documents",
                    file_types = [".pdf"],
                    file_count = "multiple"
                )
                upload_btn = gr.Button("Process PDFs", variant = "secondary")
                upload_status = gr.Textbox(
                    label = "Upload Status",
                    interactive = False,
                    lines = 3
                )
        
        # Right Column - Downloads
        with gr.Column(scale = 1):
            gr.Markdown("### 📥 Generated Documents")
            
            file_output = gr.File(
                label = "Available Documents",
                file_count = "multiple",
                interactive = False
            )
            
            refresh_btn = gr.Button("🔄 Refresh Document List", variant = "secondary")
            
            gr.Markdown("""
            **Tips:**
            - Ask me to create documents and they'll appear here
            - Click any document to download it
            - Documents are saved in the `documents/` folder
            """)
    
    # Event handlers
    def handle_send(message, history):
        """Handle sending messages."""
        if message.strip():
            # Add user message to history
            history.append({"role": "user", "content": message})
            
            # Get AI response
            ai_response = process_message(message, history)
            
            # Add AI response to history
            history.append({"role": "assistant", "content": ai_response})
            
            # Refresh document list in case new documents were created
            available_docs = list_documents()
            
            return history, "", available_docs
        
        return history, message, None
    
    def handle_upload(files):
        """Handle file upload."""
        status = upload_pdfs(files)
        return status
    
    def handle_refresh():
        """Refresh the document list."""
        return list_documents()
    
    # Connect events
    send_btn.click(
        handle_send,
        inputs = [msg, chatbot],
        outputs = [chatbot, msg, file_output]
    )
    
    msg.submit(
        handle_send,
        inputs = [msg, chatbot],
        outputs = [chatbot, msg, file_output]
    )
    
    upload_btn.click(
        handle_upload,
        inputs = [file_upload],
        outputs = [upload_status]
    )
    
    refresh_btn.click(
        handle_refresh,
        outputs = [file_output]
    )
    
    # Initial load of documents
    demo.load(
        handle_refresh,
        outputs = [file_output]
    )


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is required.")
        print("Please set your Groq API key in your .env file:")
        print("GROQ_API_KEY=your-groq-api-key-here")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY environment variable is required.")
        print("Please set your Tavily API key in your .env file:")
        print("TAVILY_API_KEY=your-tavily-api-key-here")
        exit(1)
    
    print("✅ Starting AI Assistant...")
    
    # Launch the Gradio app
    demo.launch(
        share = False,
        show_error = True
    )
