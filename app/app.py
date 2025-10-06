"""
Job Application Assistant using Gradio and LangGraph

This application provides a web-based chatbot interface for job applications with:
1. Resume and job posting upload/download capabilities
2. Side panel showing available documents
3. Flexible document reading (PDFs for resumes, web search for job postings)
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
        # Add user message to history
        history.append([message, None])
        
        # Run the agent
        config = {"configurable": {"thread_id": "gradio_session"}}
        
        response = agent_app.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "resumes": conversation_state["resumes"],
                "job_summaries": conversation_state["job_summaries"]
            },
            config=config
        )
        
        # Update conversation state
        if "resumes" in response:
            conversation_state["resumes"].update(response["resumes"])
        if "job_summaries" in response:
            conversation_state["job_summaries"].update(response["job_summaries"])
        
        # Get the last AI message
        last_message = response["messages"][-1]
        
        if isinstance(last_message, AIMessage):
            ai_response = last_message.content
            history[-1][1] = ai_response
            
            # Update side panel
            return history, get_available_documents()
        
        return history, get_available_documents()
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history[-1][1] = error_msg
        return history, get_available_documents()

def upload_resume(file):
    """Handle resume upload."""
    if file is None:
        return "No file uploaded"
    
    try:
        # Save uploaded file temporarily
        temp_path = file.name
        filename = os.path.basename(temp_path)
        
        # Read the document
        result = pdf_reader.invoke({"filepath": temp_path})
        
        # Store in conversation state
        conversation_state["resumes"][filename] = {
            "content": result,
            "filename": filename,
            "filepath": temp_path
        }
        
        return f"✅ Document '{filename}' uploaded successfully!"
        
    except Exception as e:
        return f"❌ Error uploading document: {str(e)}"

def upload_job_posting(file):
    """Handle job posting upload (PDF)."""
    if file is None:
        return "No file uploaded"
    
    try:
        # Save uploaded file temporarily
        temp_path = file.name
        filename = os.path.basename(temp_path)
        
        # For PDF documents, we'll use pdf_reader to extract text
        # then treat it as content
        result = pdf_reader.invoke({"filepath": temp_path})
        
        # Store in conversation state as job posting
        conversation_state["job_summaries"][filename] = {
            "query": filename,
            "content": result,
            "added_at": "now"
        }
        
        return f"✅ Content '{filename}' uploaded successfully!"
        
    except Exception as e:
        return f"❌ Error uploading content: {str(e)}"

def search_job_online(query):
    """Search for job postings online."""
    if not query.strip():
        return "Please enter a search query"
    
    try:
        result = web_search.invoke({"query_or_url": query})
        
        # Store in conversation state
        conversation_state["job_summaries"][query] = {
            "query": query,
            "content": result,
            "added_at": "now"
        }
        
        return f"✅ Found content for: {query}"
        
    except Exception as e:
        return f"❌ Error searching for content: {str(e)}"

def get_available_documents():
    """Get list of available resumes and job postings."""
    resume_list = []
    job_list = []
    
    for filename, data in conversation_state["resumes"].items():
        resume_list.append(f"📄 {filename}")
    
    for query, data in conversation_state["job_summaries"].items():
        job_list.append(f"💼 {query}")
    
    resume_text = "\n".join(resume_list) if resume_list else "No documents available"
    job_text = "\n".join(job_list) if job_list else "No searched content available"
    
    return resume_text, job_text

def download_cover_letter(resume_filename, job_query, create_doc=True):
    """Create and download a cover letter."""
    if not resume_filename or not job_query:
        return "Please select both resume and job posting"
    
    try:
        result = create_cover_letter.invoke({
            "resume_filename": resume_filename,
            "job_query": job_query,
            "create_document": create_doc
        })
        
        return result
        
    except Exception as e:
        return f"❌ Error creating cover letter: {str(e)}"

def clear_conversation():
    """Clear the conversation and uploaded documents."""
    global conversation_state
    conversation_state = {"messages": [], "resumes": {}, "job_summaries": {}}
    return [], "No documents available", "No searched content available"

# Create Gradio interface
with gr.Blocks(title="Job Application Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Job Application Assistant")
    gr.Markdown("Upload documents and search for content, then chat with the AI to create cover letters!")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Chat with the Assistant",
                height=400,
                show_copy_button=True,
                type='messages'
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything about job applications...",
                    label="Your Message",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Upload sections
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📄 Upload Document")
                    resume_upload = gr.File(
                        label="Upload PDF Document",
                        file_types=[".pdf"],
                        file_count="single"
                    )
                    resume_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 💼 Upload Content")
                    job_upload = gr.File(
                        label="Upload PDF Content",
                        file_types=[".pdf"],
                        file_count="single"
                    )
                    job_status = gr.Textbox(label="Status", interactive=False)
            
            # Online content search
            gr.Markdown("### 🔍 Search Content Online")
            with gr.Row():
                job_query = gr.Textbox(
                    placeholder="Enter search query, company, or URL...",
                    label="Search Query",
                    scale=3
                )
                search_btn = gr.Button("Search", variant="secondary", scale=1)
            search_status = gr.Textbox(label="Search Status", interactive=False)
        
        with gr.Column(scale=1):
            # Side panel for available documents
            gr.Markdown("### 📋 Available Documents")
            
            gr.Markdown("**Documents:**")
            resumes_display = gr.Textbox(
                label="",
                lines=8,
                interactive=False,
                value="No documents available"
            )
            
            gr.Markdown("**Searched Content:**")
            jobs_display = gr.Textbox(
                label="",
                lines=8,
                interactive=False,
                value="No searched content available"
            )
            
            # Cover letter creation
            gr.Markdown("### ✍️ Create Cover Letter")
            with gr.Column():
                resume_select = gr.Dropdown(
                    choices=[],
                    label="Select Document",
                    interactive=True
                )
                job_select = gr.Dropdown(
                    choices=[],
                    label="Select Content",
                    interactive=True
                )
                create_doc_checkbox = gr.Checkbox(
                    label="Create Word Document",
                    value=True
                )
                cover_letter_btn = gr.Button("Create Cover Letter", variant="primary")
                cover_letter_result = gr.Textbox(
                    label="Result",
                    lines=5,
                    interactive=False
                )
            
            # Clear button
            clear_btn = gr.Button("🗑️ Clear All", variant="stop")
    
    # Event handlers
    def update_dropdowns():
        """Update dropdown choices based on available documents."""
        resume_choices = list(conversation_state["resumes"].keys())
        job_choices = list(conversation_state["job_summaries"].keys())
        return gr.Dropdown(choices=resume_choices), gr.Dropdown(choices=job_choices)
    
    # Chat functionality
    def handle_send(message, history):
        """Handle sending messages."""
        if message.strip():
            return process_message(message, history)
        return history, get_available_documents()
    
    # File upload handlers
    def handle_resume_upload(file):
        """Handle resume upload."""
        status = upload_resume(file)
        resume_display, job_display = get_available_documents()
        resume_choices, job_choices = update_dropdowns()
        return status, resume_display, job_display, resume_choices, job_choices
    
    def handle_job_upload(file):
        """Handle job posting upload."""
        status = upload_job_posting(file)
        resume_display, job_display = get_available_documents()
        resume_choices, job_choices = update_dropdowns()
        return status, resume_display, job_display, resume_choices, job_choices
    
    def handle_job_search(query):
        """Handle online job search."""
        status = search_job_online(query)
        resume_display, job_display = get_available_documents()
        resume_choices, job_choices = update_dropdowns()
        return status, resume_display, job_display, resume_choices, job_choices
    
    def handle_cover_letter_creation(resume, job, create_doc):
        """Handle cover letter creation."""
        result = download_cover_letter(resume, job, create_doc)
        return result
    
    def handle_clear():
        """Handle clearing conversation."""
        cleared_state = clear_conversation()
        resume_choices, job_choices = update_dropdowns()
        return cleared_state[0], cleared_state[1], cleared_state[2], resume_choices, job_choices
    
    # Connect events
    send_btn.click(
        handle_send,
        inputs=[msg, chatbot],
        outputs=[chatbot, resumes_display, jobs_display]
    )
    
    msg.submit(
        handle_send,
        inputs=[msg, chatbot],
        outputs=[chatbot, resumes_display, jobs_display]
    )
    
    resume_upload.change(
        handle_resume_upload,
        inputs=[resume_upload],
        outputs=[resume_status, resumes_display, jobs_display, resume_select, job_select]
    )
    
    job_upload.change(
        handle_job_upload,
        inputs=[job_upload],
        outputs=[job_status, resumes_display, jobs_display, resume_select, job_select]
    )
    
    search_btn.click(
        handle_job_search,
        inputs=[job_query],
        outputs=[search_status, resumes_display, jobs_display, resume_select, job_select]
    )
    
    cover_letter_btn.click(
        handle_cover_letter_creation,
        inputs=[resume_select, job_select, create_doc_checkbox],
        outputs=[cover_letter_result]
    )
    
    clear_btn.click(
        handle_clear,
        outputs=[chatbot, resumes_display, jobs_display, resume_select, job_select]
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
        # server_name="0.0.0.0",
        # server_port=7860,
        share=False,
        show_error=True
    )