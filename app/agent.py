"""
LangGraph Agent for General Purpose AI Assistant

This module contains the LangGraph workflow that coordinates:
- Tool selection and execution
- Conversation management
- State handling
"""

import os
from typing import Annotated, Literal, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from tools import get_all_tools, process_and_store_pdf
from datetime import date


# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    groq_api_key = os.getenv("GROQ_API_KEY")
)

# Get tools
tools = get_all_tools()
llm_with_tools = llm.bind_tools(tools)

# Initialize memory for conversation
memory = MemorySaver()


class AgentState(TypedDict):
    """State for the general purpose agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def agent_node(state: AgentState):
    """Agent node that processes messages and decides on actions."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"


# Create the tool node
tool_node = ToolNode(tools)

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

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
app = workflow.compile(checkpointer = memory)


def get_system_message() -> str:
    """Get the system message for the agent."""
    return f"""You are a helpful AI assistant with access to two powerful tools:

1. **retrieve_documents**: Use this to search through uploaded documents in the knowledge base. 
   - Use when users ask about content from uploaded PDFs or documents
   - Performs semantic search to find relevant information

2. **web_search**: Use this to search the internet for current information.
   - Use for current events, recent news, or information not in the documents
   - Provides real-time web search results

Guidelines:
- Always be helpful and conversational
- If you can't answer by yourself, use retrieve_documents first.
- If you can't find it in the documents, use web_search.
- If you can't find it in both, ask the user to provide more information.
- Explain your actions clearly to the user
- The current date is {date.today().strftime("%b %d, %Y")}
"""

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
    
    # Process PDF
    filepath = 'Resume.pdf'
    print(f"\n📄 Processing PDF: {filepath}")
    num_chunks = process_and_store_pdf(filepath)
    print(f"✅ Processed {num_chunks} chunks from {filepath}\n")
    
    # Interactive CLI loop
    print("🤖 General Purpose AI Assistant")
    print("=" * 50)
    print("I can help you with:")
    print("• Searching through uploaded documents")
    print("• Searching the web for current information")
    print("\nType 'exit' to quit.")
    print("=" * 50)
    
    config = {"configurable": {"thread_id": "cli_session"}}
    
    while True:
        user_input = input("\n👤 You:  ")
        
        if user_input.strip().lower() == 'exit':
            print("👋  Goodbye!")
            break
        
        if not user_input.strip():
            continue
        
        try:
            # Run the agent
            from langchain_core.messages import HumanMessage
            
            response = app.invoke(
                {"messages": [HumanMessage(content = user_input.strip())]},
                config = config
            )
            
            # Get the last AI message
            last_message = response["messages"][-1]
            print(f"\n🤖 Assistant: {last_message.content}")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or rephrase your question.")
    
