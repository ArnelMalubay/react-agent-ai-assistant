"""
Gradio Interface for General Purpose AI Assistant

This module provides a web interface with:
- Chat interface for conversing with the AI agent
- PDF upload functionality for RAG
- File list showing all uploaded documents
"""

import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import app as agent_app, get_system_message
from tools import process_and_store_pdf

# Load environment variables
load_dotenv()

# Global configuration
COLLECTION_NAME = "general_collection"
CHROMA_DIR = None  # Set to None for ephemeral (in-memory) storage, or "./chroma_db" for persistent storage

# Track uploaded files
uploaded_files = []


def process_message(message, history):
    """Process user message through the agent and show tool usage in real-time."""
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
        
        # Stream the agent response
        tool_indicators = []
        
        for event in agent_app.stream(
            {"messages": all_messages},
            config = config
        ):
            # Check if this is an agent node output
            if "agent" in event:
                agent_msg = event["agent"]["messages"][-1]
                
                # Check for tool calls
                if isinstance(agent_msg, AIMessage) and hasattr(agent_msg, 'tool_calls') and agent_msg.tool_calls:
                    for tool_call in agent_msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        if tool_name == 'retrieve_documents':
                            indicator = "🔍 Searching through uploaded documents..."
                            if indicator not in tool_indicators:
                                tool_indicators.append(indicator)
                                yield "\n".join(tool_indicators)
                        elif tool_name == 'web_search':
                            indicator = "🌐 Searching the web..."
                            if indicator not in tool_indicators:
                                tool_indicators.append(indicator)
                                yield "\n".join(tool_indicators)
                
                # Check for final response
                if isinstance(agent_msg, AIMessage) and agent_msg.content and not agent_msg.tool_calls:
                    # This is the final response
                    if tool_indicators:
                        final_response = "\n".join(tool_indicators) + "\n\n" + agent_msg.content
                    else:
                        final_response = agent_msg.content
                    yield final_response
                    return
        
    except Exception as e:
        # Check if it's a tool use error
        error_str = str(e)
        if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
            yield "I apologize, but I encountered an issue while trying to process your request. Could you please rephrase your question or provide more details?"
        else:
            yield "I'm having trouble processing that request right now. Please try asking in a different way."


def upload_pdfs(files):
    """Handle PDF uploads and add to Chroma collection."""
    if not files:
        return "No files uploaded.", get_file_list()
    
    try:
        processed_files = []
        total_chunks = 0
        
        for file in files:
            # Use the LangChain-based utility function from tools.py
            num_chunks = process_and_store_pdf(
                filepath = file.name,
                collection_name = COLLECTION_NAME,
                chunk_size = 500,
                chunk_overlap = 150,
                persist_directory = CHROMA_DIR
            )
            
            if num_chunks > 0:
                filename = os.path.basename(file.name)
                if filename not in uploaded_files:
                    uploaded_files.append(filename)
                processed_files.append(filename)
                total_chunks += num_chunks
        
        if processed_files:
            file_list = ", ".join(processed_files)
            status = f"✅ Successfully processed {len(processed_files)} file(s): {file_list}\n📦 Added {total_chunks} chunks to the knowledge base."
        else:
            status = "❌ No files were successfully processed."
        
        return status, get_file_list()
            
    except Exception as e:
        return f"❌ Error processing files: {str(e)}", get_file_list()


def get_file_list():
    """Get formatted list of uploaded files."""
    if not uploaded_files:
        return "No files uploaded yet"
    
    file_list = []
    for i, filename in enumerate(uploaded_files, 1):
        file_list.append(f"{i}. {filename}")
    
    return "\n".join(file_list)


def clear_files():
    """Clear the uploaded files list."""
    global uploaded_files
    uploaded_files = []
    return "No files uploaded yet"


with gr.Blocks(title = "PDF Explainer Chatbot") as demo:
    gr.Markdown("# 🤖 ReAct Agent Assistant")
    gr.Markdown("""
    **I'm an AI assistant built using the ReAct agentic framework. I am capable of answering general questions, analyze uploaded PDF documents (through RAG), and perform web searches.**

    - 📤 **Upload PDFs**: Add documents anytime to get document-specific answers. Press Process PDFs below to add documents to my knowledge base.
    - 🌐 **Search the Web**: I can perform web searches to get the latest information and news.
    """)
    
    chatbot = gr.Chatbot(
        height = 500,
        show_copy_button = True,
        type = 'messages',
        value = [{"role": "assistant", "content": "Hello! I'm here to help you with any questions or tasks you have. Ask away or upload PDFs if you want. I'm ready when you are!"}]
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder = "Type your message here...",
            show_label = False,
            scale = 4
        )
        send_btn = gr.Button("Send", variant = "primary", scale = 1)
    
    with gr.Row():
        file_upload = gr.File(
            label = "Upload PDF Documents",
            file_types = [".pdf"],
            file_count = "multiple"
        )
        process_btn = gr.Button("Process PDFs", variant = "secondary")
    
    upload_status = gr.Textbox(
        label = "Upload Status",
        interactive = False,
        lines = 2
    )
    
    # Event handlers
    def handle_send(message, history):
        """Handle sending messages with streaming."""
        if message.strip():
            # Add user message to history
            history.append({"role": "user", "content": message})
            
            # Stream AI response with tool indicators
            for partial_response in process_message(message, history):
                # Update the assistant's message in real-time
                if len(history) > 0 and history[-1]["role"] == "assistant":
                    history[-1]["content"] = partial_response
                else:
                    history.append({"role": "assistant", "content": partial_response})
                
                yield history, ""
            
            return
        
        yield history, message
    
    def handle_upload(files):
        """Handle file upload."""
        status, file_list = upload_pdfs(files)
        return status
    
    # Connect events
    send_btn.click(
        handle_send,
        inputs = [msg, chatbot],
        outputs = [chatbot, msg]
    )
    
    msg.submit(
        handle_send,
        inputs = [msg, chatbot],
        outputs = [chatbot, msg]
    )
    
    process_btn.click(
        handle_upload,
        inputs = [file_upload],
        outputs = [upload_status]
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
        show_error = True,
        server_name = "0.0.0.0", 
        server_port = 7860
    )
