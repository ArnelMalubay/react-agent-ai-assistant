"""
Job Application Assistant using LangGraph and LangChain

This application provides an AI agent that helps with job applications by:
1. Reading and analyzing resumes
2. Searching for job postings
3. Creating cover letters and documents
"""

import os
from typing import Annotated, Literal, Sequence, TypedDict
from pathlib import Path
from dotenv import load_dotenv

import re, unicodedata
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyMuPDFLoader
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from docx import Document
from docx.shared import Inches

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Tavily search
tavily_search = TavilySearch(
    max_results = 1, 
    topic = 'general', 
    search_depth = 'basic',
    include_answer = 'advanced'
)

# Initialize memory for conversation
memory = MemorySaver()


class AgentState(TypedDict):
    """State for the job application agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    resumes: dict  # Dictionary to store resume data by filename/id
    job_summaries: dict  # Dictionary to store job posting data by query/url


def add_resume_to_state(state: AgentState, filename: str, content: str) -> AgentState:
    """Add resume content to the state."""
    if "resumes" not in state:
        state["resumes"] = {}
    state["resumes"][filename] = content
    return state

def add_job_to_state(state: AgentState, query: str, results: dict) -> AgentState:
    """Add job posting data to the state."""
    if "job_summaries" not in state:
        state["job_summaries"] = {}
    state["job_summaries"][query] = results
    return state

def get_resume_options(state: AgentState) -> str:
    """Get formatted list of available resumes."""
    if "resumes" not in state or not state["resumes"]:
        return "No resumes available."
    
    options = []
    for i, (filename, data) in enumerate(state["resumes"].items(), 1):
        options.append(f"{i}. {filename}")
    
    return "\n".join(options)


def get_job_options(state: AgentState) -> str:
    """Get formatted list of available job postings."""
    if "job_summaries" not in state or not state["job_summaries"]:
        return "No job postings available."
    
    options = []
    for i, (query, data) in enumerate(state["job_summaries"].items(), 1):
        options.append(f"{i}. {query}")
    
    return "\n".join(options)


def clean_text(text: str) -> str:
    """
    Clean text for better performance while preserving markdown structure.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text optimized for embedding and chunking
    """
    if not text or not text.strip():
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Fix common PDF extraction artifacts
    # Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove excessive whitespace while preserving structure
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\t+', ' ', text)  # Tabs to single space
    text = re.sub(r'\n +', '\n', text)  # Remove spaces after newlines
    text = re.sub(r' +\n', '\n', text)  # Remove spaces before newlines
    
    # Normalize line breaks (preserve paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'\r\n', '\n', text)  # Windows line endings to Unix
    text = re.sub(r'\r', '\n', text)  # Old Mac line endings to Unix
    
    # Clean up common PDF artifacts
    # Remove standalone page numbers (numbers on their own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove standalone roman numerals (common in headers/footers)
    text = re.sub(r'\n\s*[ivxlcdm]+\s*\n', '\n', text, flags = re.IGNORECASE)
    
    # Clean up markdown table formatting (preserve structure but clean spacing)
    # Fix spacing around table delimiters
    text = re.sub(r' +\| +', ' | ', text)  # Normalize spacing around pipes
    text = re.sub(r'^\| +', '| ', text, flags = re.MULTILINE)  # Start of line pipes
    text = re.sub(r' +\|$', ' |', text, flags = re.MULTILINE)  # End of line pipes
    
    # Preserve list formatting but clean spacing
    text = re.sub(r'\n +([•\-\*\+])', r'\n\1', text)  # Bullet lists
    text = re.sub(r'\n +(\d+\.)', r'\n\1', text)  # Numbered lists
    
    # Clean up header formatting (preserve markdown headers)
    text = re.sub(r'\n +(#+)', r'\n\1', text)  # Remove spaces before headers
    text = re.sub(r'(#+) +([^\n]+)', r'\1 \2', text)  # Normalize header spacing
    
    # Remove excessive punctuation (but preserve meaningful punctuation)
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
    text = re.sub(r'-{3,}', '---', text)  # Multiple dashes to em dash
    
    # Clean up quote marks
    text = re.sub(r'[\u201C\u201D\u201E]', '"', text)  # Normalize quotes
    text = re.sub(r'[\u2018\u2019]', "'", text)  # Normalize apostrophes
    
    # Remove zero-width characters and other invisible characters
    text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)
    
    # Final cleanup
    text = text.strip()  # Remove leading/trailing whitespace
    
    # Ensure text doesn't start or end with newlines after cleaning
    text = text.strip('\n')
    
    return text

@tool
def resume_reader(filepath: str) -> str:
    """
    Read a resume from a PDF file and return its text content.
    The resume will be stored in the agent's memory for future reference.
    
    Args:
        filepath: Path to the PDF resume file
        
    Returns:
        str: The text content of the resume
    """
    try:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' not found."
        
        if not filepath.lower().endswith('.pdf'):
            return "Error: Only PDF files are supported for resume reading."
        
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        
        if not documents:
            return "Error: Could not extract text from the PDF file."
        
        # Combine all pages into a single text
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Extract filename for storage
        filename = os.path.basename(filepath)
        
        return f"Successfully read resume '{filename}'. Content preview:\n{clean_text(text_content)}"
    
    except Exception as e:
        return f"Error reading resume: {str(e)}"


@tool
def job_search(query_or_url: str) -> dict:
    """
    Search for job postings using Tavily search.
    
    Args:
        query_or_url: Either a search query or a URL to a job posting
        
    Returns:
        dict: Search results or extracted content
    """
    try:
        results = tavily_search.invoke(query_or_url)
        
        return {
            "query": query_or_url,
            "answer": results['answer'],
            "status": "success"
        }
    
    except Exception as e:
        return {
            "query": query_or_url,
            "answer": '',
            "status": "error",
            "error": str(e)
        }


@tool
def list_resumes() -> str:
    """
    List all available resumes in the agent's memory.
    
    Returns:
        str: Formatted list of available resumes
    """
    return "This tool will list available resumes. The agent should check its state for resume data."


@tool
def list_job_postings() -> str:
    """
    List all available job postings in the agent's memory.
    
    Returns:
        str: Formatted list of available job postings
    """
    return "This tool will list available job postings. The agent should check its state for job data."


@tool
def create_cover_letter(resume_filename: str, job_query: str, create_document: bool = True) -> str:
    """
    Create a cover letter based on a specific resume and job posting.
    
    Args:
        resume_filename: The filename of the resume to use
        job_query: The query/URL of the job posting to use
        create_document: Whether to create a Word document (default: True)
        
    Returns:
        str: The cover letter content or document creation status
    """
    return f"This tool will create a cover letter using resume '{resume_filename}' and job '{job_query}'. Create document: {create_document}"


@tool
def document_creator(content: str, filename: str = "cover_letter.docx") -> str:
    """
    Create a Word document with the given content.
    
    Args:
        content: The text content to include in the document
        filename: Name of the output file (default: cover_letter.docx)
        
    Returns:
        str: Path to the created document
    """
    try:
        # Ensure we have a .docx extension
        if not filename.endswith('.docx'):
            filename += '.docx'
        
        # Create documents directory if it doesn't exist
        docs_dir = Path("documents")
        docs_dir.mkdir(exist_ok=True)
        
        filepath = docs_dir / filename
        
        # Create a new Document
        doc = Document()
        
        # Add content
        doc.add_paragraph(content)
        
        # Add some spacing
        doc.add_paragraph()
        
        # Save the document
        doc.save(str(filepath))
        
        return f"Document created successfully at: {filepath}"
    
    except Exception as e:
        return f"Error creating document: {str(e)}"


# Bind tools to the LLM
tools = [resume_reader, job_search, list_resumes, list_job_postings, create_cover_letter, document_creator]
llm_with_tools = llm.bind_tools(tools)

# Create the agent node
def agent_node(state: AgentState):
    """Agent node that processes messages and decides on actions."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Create the conditional routing function
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"


# Create a custom tool node that can update state
def custom_tool_node(state: AgentState):
    """Custom tool node that can update state based on tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Execute the tool
        if tool_name == "resume_reader":
            result = resume_reader.invoke(tool_args)
            
            # Extract filename and add to state
            filepath = tool_args.get("filepath", "")
            filename = os.path.basename(filepath)
            
            # Load the actual content for storage
            try:
                if os.path.exists(filepath) and filepath.lower().endswith('.pdf'):
                    loader = PyMuPDFLoader(filepath)
                    documents = loader.load()
                    text_content = "\n\n".join([doc.page_content for doc in documents])
                    
                    # Update state with resume data
                    if "resumes" not in state:
                        state["resumes"] = {}
                    state["resumes"][filename] = {
                        "content": text_content,
                        "filename": filename,
                        "filepath": filepath
                    }
            except Exception as e:
                pass  # Continue with the tool result even if state update fails
            
        elif tool_name == "job_search":
            result = job_search.invoke(tool_args)
            
            # Add to state
            query = tool_args.get("query_or_url", "")
            if "job_summaries" not in state:
                state["job_summaries"] = {}
            state["job_summaries"][query] = {
                "query": query,
                "results": result,
                "added_at": "now"
            }
            
        elif tool_name == "list_resumes":
            if "resumes" in state and state["resumes"]:
                result = "Available resumes:\n" + "\n".join([f"• {filename}" for filename in state["resumes"].keys()])
            else:
                result = "No resumes available. Please read a resume first."
                
        elif tool_name == "list_job_postings":
            if "job_summaries" in state and state["job_summaries"]:
                result = "Available job postings:\n" + "\n".join([f"• {query}" for query in state["job_summaries"].keys()])
            else:
                result = "No job postings available. Please search for job postings first."
                
        elif tool_name == "create_cover_letter":
            resume_filename = tool_args.get("resume_filename", "")
            job_query = tool_args.get("job_query", "")
            
            # Check if resume and job exist in state
            if "resumes" not in state or resume_filename not in state["resumes"]:
                result = f"Error: Resume '{resume_filename}' not found in memory. Please read the resume first."
            elif "job_summaries" not in state or job_query not in state["job_summaries"]:
                result = f"Error: Job posting '{job_query}' not found in memory. Please search for the job posting first."
            else:
                # Generate cover letter using LLM
                resume_content = state["resumes"][resume_filename]["content"]
                job_data = state["job_summaries"][job_query]["results"]
                
                # Create a prompt for cover letter generation
                prompt = f"""Create a professional cover letter based on the following resume and job posting:

RESUME CONTENT:
{resume_content[:2000]}

JOB POSTING DATA:
{str(job_data)[:2000]}

Please create a compelling cover letter that:
1. Highlights relevant experience from the resume
2. Addresses key requirements from the job posting
3. Shows enthusiasm for the position
4. Is professional and well-structured

Cover Letter:"""
                
                # Use LLM to generate cover letter
                response = llm.invoke([HumanMessage(content=prompt)])
                cover_letter = response.content
                
                # Check if user wants a document
                create_doc = tool_args.get("create_document", True)
                if create_doc:
                    # Create document
                    doc_result = document_creator.invoke({"content": cover_letter, "filename": f"cover_letter_{resume_filename}_{job_query.replace('/', '_')}.docx"})
                    result = f"Cover letter created successfully!\n\n{cover_letter}\n\n{doc_result}"
                else:
                    result = f"Cover letter created:\n\n{cover_letter}"
                    
        elif tool_name == "document_creator":
            result = document_creator.invoke(tool_args)
            
        else:
            result = f"Tool {tool_name} not implemented"
        
        # Create tool message
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        tool_messages.append(tool_message)
    
    return {"messages": tool_messages}

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", custom_tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile(checkpointer=memory)


def run_job_assistant():
    """Run the job application assistant."""
    print("🤖 Job Application Assistant")
    print("=" * 50)
    print("I can help you with:")
    print("• Reading and analyzing resumes")
    print("• Searching for job postings")
    print("• Creating cover letters")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("=" * 50)
    
    # Initialize conversation with system message
    config = {"configurable": {"thread_id": "job_assistant_session"}}
    
    # Send initial system message
    system_message = """You are a job application assistant. Your key behaviors:

1. When asked to create a cover letter:
   - First check if resumes and job postings are available in memory
   - If missing, ask the user to provide a resume file path and/or job posting URL/query
   - Confirm which specific resume and job posting combination to use
   - Ask if they want a Word document created

2. When reading resumes:
   - Use the resume_reader tool with the file path
   - The resume will be stored in memory for future use

3. When searching for jobs:
   - Use the job_search tool with URL or query
   - The job posting will be stored in memory for future use

4. Always be helpful and guide users through the process step by step."""
    
    # Initialize with system message
    app.invoke(
        {
            "messages": [HumanMessage(content=system_message)],
            "resumes": {},
            "job_summaries": {}
        },
        config=config
    )
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye! Good luck with your job applications!")
            break
        
        if not user_input:
            continue
        
        try:
            # Run the agent with proper state initialization
            response = app.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "resumes": {},
                    "job_summaries": {}
                },
                config=config
            )
            
            # Get the last AI message
            last_message = response["messages"][-1]
            
            if isinstance(last_message, AIMessage):
                print(f"\n🤖 Assistant: {last_message.content}")
                
                # If there were tool calls, show their results
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print("\n🔧 Tool Results:")
                    for tool_call in last_message.tool_calls:
                        print(f"  • {tool_call['name']}: {tool_call['args']}")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or rephrase your question.")


# if __name__ == "__main__":
#     # Check for required environment variables
#     if not os.getenv("GROQ_API_KEY"):
#         print("❌ Error: GROQ_API_KEY environment variable is required.")
#         print("Please set your Groq API key in your .env file:")
#         print("GROQ_API_KEY=your-groq-api-key-here")
#         exit(1)
    
#     if not os.getenv("TAVILY_API_KEY"):
#         print("❌ Error: TAVILY_API_KEY environment variable is required.")
#         print("Please set your Tavily API key in your .env file:")
#         print("TAVILY_API_KEY=your-tavily-api-key-here")
#         exit(1)
    
#     run_job_assistant()

# filepath = 'app/Arnel Malubay Resume.pdf'
# print(resume_reader.invoke(filepath))

# query_or_url = 'https://thinkingmachines.freshteam.com/jobs/hlJrAwsfhoF2/ph-business-intelligence-analyst'
# print(job_search.invoke(query_or_url))

# content = '''
# Sample Content
# Sample Content
# '''
# print(document_creator.invoke(content))

# TO DO: Double check tools; maybe remove state management??