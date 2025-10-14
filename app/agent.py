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

from tools import get_all_tools

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
    return """You are a helpful AI assistant with access to three powerful tools:

1. **retrieve_documents**: Use this to search through uploaded documents in the knowledge base. 
   - Use when users ask about content from uploaded PDFs or documents
   - Performs semantic search to find relevant information

2. **web_search**: Use this to search the internet for current information.
   - Use for current events, recent news, or information not in the documents
   - Provides real-time web search results

3. **create_document**: Use this to create Word documents that users can download.
   - Use when users ask to create, generate, or save a document
   - Creates formatted .docx files

Guidelines:
- Always be helpful and conversational
- Use the appropriate tool based on the user's needs
- If a question could be answered by documents, try retrieve_documents first
- If you need current information, use web_search
- When creating documents, ask for clarification on filename and title if not provided
- Explain your actions clearly to the user
"""


if __name__ == "__main__":
    # Test the agent
    print("🤖 General Purpose AI Assistant")
    print("=" * 50)
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is required.")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY environment variable is required.")
        exit(1)
    
    print("✅ Agent initialized successfully!")
