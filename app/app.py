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


# Custom CSS for clean, professional design matching the reference image
custom_css = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    max-width: 1200px !important;
    margin: auto !important;
}

.contain {
    background: #f8fafc !important;
}

h1 {
    color: #1e293b !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    margin-bottom: 1rem !important;
}

h2 {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    margin: 2rem 0 1rem 0 !important;
}

#chatbot {
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    background: white !important;
    border: 1px solid #e5e7eb !important;
}

.message {
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 8px 0 !important;
}

.user {
    background: #eff6ff !important;
    border-left: 4px solid #3b82f6 !important;
}

.bot {
    background: #f9fafb !important;
    border-left: 4px solid #6b7280 !important;
}

textarea {
    border-radius: 8px !important;
    border: 2px solid #d1d5db !important;
    font-size: 15px !important;
    background: white !important;
}

textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

.primary {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    color: white !important;
}

.primary:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

.secondary {
    background: white !important;
    border: 2px solid #3b82f6 !important;
    color: #3b82f6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.secondary:hover {
    background: #eff6ff !important;
}

.markdown {
    color: #4b5563 !important;
    line-height: 1.6 !important;
}

/* Upload area styling */
.wrap {
    border: 2px dashed #d1d5db !important;
    border-radius: 8px !important;
    background: #f9fafb !important;
    padding: 20px !important;
}

.wrap:hover {
    border-color: #3b82f6 !important;
    background: #eff6ff !important;
}

/* File upload button styling */
input[type="file"] {
    border-radius: 8px !important;
    padding: 10px !important;
}

/* Status text styling */
.textbox {
    background: #f3f4f6 !important;
    border-radius: 6px !important;
    border: 1px solid #e5e7eb !important;
}
"""

# Create Gradio interface with custom theme
theme = gr.themes.Base(
    primary_hue = "blue",
    secondary_hue = "slate",
    neutral_hue = "slate",
    font = [gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono = [gr.themes.GoogleFont("Fira Code"), "monospace"]
)

with gr.Blocks(title = "PDF Explainer Chatbot", theme = theme, css = custom_css) as demo:
    gr.Markdown("""
    # 📄 PDF Explainer Chatbot
    
    I'm an AI assistant that can help you with general questions and analyze PDF documents you upload.
    
    **Key Features:**
    • 💬 **Chat normally:** Ask me anything, even without uploading PDFs
    • 📤 **Upload PDFs:** Add documents anytime to get document-specific answers  
    • 📚 **Multiple uploads:** You can upload more PDFs during our conversation
    • 🔍 **Smart retrieval:** I'll automatically find relevant content from your PDFs when answering questions
    """)
    
    # PDF Upload Section at the top
    with gr.Row():
        upload_col1 = gr.Column(scale = 1)
        upload_col2 = gr.Column(scale = 3)
        upload_col3 = gr.Column(scale = 1)
        
        with upload_col1:
            gr.Markdown("")
            
        with upload_col2:
            with gr.Row():
                upload_btn_label = gr.Markdown("**Upload PDF Documents (Optional)**")
                file_upload = gr.File(
                    label = "",
                    file_types = [".pdf"],
                    file_count = "multiple",
                    container = False,
                    scale = 4
                )
                process_btn = gr.Button("📄 Process PDFs", variant = "primary", scale = 1)
        
        with upload_col3:
            gr.Markdown("")
    
    upload_status = gr.Textbox(
        show_label = False,
        interactive = False,
        lines = 2,
        container = False,
        visible = True
    )
    
    # Chat Section below
    gr.Markdown("""
    ---
    ## 💬 Chat
    
    Ask me anything! If you've uploaded PDFs, I'll use them to provide more accurate answers.
    """)
    
    chatbot = gr.Chatbot(
        label = "",
        height = 500,
        show_copy_button = True,
        type = 'messages',
        avatar_images = (None, "🤖"),
        container = True,
        value = [{"role": "assistant", "content": "Hello! I'm here to help you with any questions or tasks you have related to PDF documents. If you'd like to get started, could you please upload the PDF documents you'd like me to assist with? This will allow me to provide more accurate and specific information. I'm ready when you are!"}]
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder = "Type your message here...",
            show_label = False,
            scale = 5,
            lines = 1,
            container = False
        )
        send_btn = gr.Button("Send", variant = "primary", scale = 1, min_width = 100)
    
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
        show_error = True
    )
