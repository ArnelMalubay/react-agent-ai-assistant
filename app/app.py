"""
Job Application Assistant using Gradio and LangGraph

This application provides a web-based chatbot interface for job applications with:
1. Document upload capabilities
2. Side panel showing available resumes and job postings
3. Simple chat interface
"""

import os
import gradio as gr
from dotenv import load_dotenv

# Import agent components
from agent import (
    app as agent_app, 
    pdf_reader, 
    web_search, 
    create_cover_letter)

from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Global state to track conversation
conversation_state = {"messages": [], "resumes": {}, "job_summaries": {}}


def process_message(message, history):
    """Process user message through the agent."""
    try:
        # Run the agent with current state
        config = {"configurable": {"thread_id": "gradio_session"}}
        
        # Create a context-aware message that includes available documents
        available_docs = ""
        if conversation_state.get("resumes"):
            available_docs += f"\n\nAvailable resumes: {', '.join(conversation_state['resumes'].keys())}"
        if conversation_state.get("job_summaries"):
            available_docs += f"\n\nAvailable job postings: {', '.join(conversation_state['job_summaries'].keys())}"
        
        # Add context to the message if there are documents
        context_message = message
        if available_docs:
            context_message = f"{message}{available_docs}"
        
        # Get all previous messages from history
        all_messages = [HumanMessage(content=context_message)]
        
        response = agent_app.invoke(
            {
                "messages": all_messages,
                "resumes": conversation_state.get("resumes", {}),
                "job_summaries": conversation_state.get("job_summaries", {})
            },
            config=config
        )
        
        # Update conversation state from response
        if "resumes" in response:
            conversation_state["resumes"] = response["resumes"]
        if "job_summaries" in response:
            conversation_state["job_summaries"] = response["job_summaries"]
        
        # Get the last AI message
        last_message = response["messages"][-1]
        
        if isinstance(last_message, AIMessage):
            ai_response = last_message.content
            return ai_response
        
        return "I'm sorry, I couldn't process that request."
        
    except Exception as e:
        return f"Error: {str(e)}"


def upload_document(file):
    """Handle document upload."""
    if file is None:
        return "No file uploaded", get_resume_list(), get_job_list()
    
    try:
        # Get filename first
        filename = os.path.basename(file.name)
        
        # Read the document using pdf_reader
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file.name)
        documents = loader.load()
        
        if not documents:
            return f"❌ Could not extract text from '{filename}'", get_resume_list(), get_job_list()
        
        # Combine all pages into a single text
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Store in conversation state with the actual content
        conversation_state["resumes"][filename] = {
            "content": text_content,
            "filename": filename,
            "filepath": file.name
        }
        
        return f"✅ Document '{filename}' uploaded successfully!", get_resume_list(), get_job_list()
        
    except Exception as e:
        return f"❌ Error uploading document: {str(e)}", get_resume_list(), get_job_list()


def get_resume_list():
    """Get formatted list of available resumes."""
    if not conversation_state["resumes"]:
        return "No resumes uploaded yet"
    
    resume_list = []
    for filename in conversation_state["resumes"].keys():
        resume_list.append(f"• {filename}")
    
    return "\n".join(resume_list)


def get_job_list():
    """Get formatted list of available job postings."""
    if not conversation_state["job_summaries"]:
        return "No job postings added yet"
    
    job_list = []
    for query in conversation_state["job_summaries"].keys():
        job_list.append(f"• {query}")
    
    return "\n".join(job_list)


def clear_all():
    """Clear all data."""
    global conversation_state
    conversation_state = {"messages": [], "resumes": {}, "job_summaries": {}}
    return None, "No resumes uploaded yet", "No job postings added yet"


# Create Gradio interface
with gr.Blocks(title="Job Application Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Job Application Assistant")
    gr.Markdown("Upload documents and chat with the AI to create cover letters!")
    
    with gr.Row():
        # Left Column - Chatbot and Upload
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat with the Assistant",
                height=500,
                show_copy_button=True,
                type='messages'
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything about job applications...",
                    label="Your Message",
                    scale=4,
                    lines=1
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                file_upload = gr.File(
                    label="📄 Upload Document (PDF)",
                    file_types=[".pdf"],
                    file_count="single"
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=1
                )
        
        # Right Column - Document Lists
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Available Documents")
            
            resumes_display = gr.Textbox(
                label="Resumes",
                lines=10,
                interactive=False,
                value="No resumes uploaded yet"
            )
            
            jobs_display = gr.Textbox(
                label="Job Postings",
                lines=10,
                interactive=False,
                value="No job postings added yet"
            )
            
            clear_btn = gr.Button("🗑️ Clear All", variant="stop")
    
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
            
            return history, "", get_resume_list(), get_job_list()
        
        return history, message, get_resume_list(), get_job_list()
    
    def handle_upload(file):
        """Handle file upload."""
        status, resume_list, job_list = upload_document(file)
        return status, resume_list, job_list
    
    def handle_clear():
        """Handle clear all."""
        return clear_all()
    
    # Connect events
    send_btn.click(
        handle_send,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, resumes_display, jobs_display]
    )
    
    msg.submit(
        handle_send,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, resumes_display, jobs_display]
    )
    
    file_upload.change(
        handle_upload,
        inputs=[file_upload],
        outputs=[upload_status, resumes_display, jobs_display]
    )
    
    clear_btn.click(
        handle_clear,
        outputs=[chatbot, resumes_display, jobs_display]
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
    
    # Launch the Gradio app
    demo.launch(
        share=False,
        show_error=True
    )
