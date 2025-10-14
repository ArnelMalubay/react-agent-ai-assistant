"""
Tools for the General Purpose AI Assistant

This module contains three main tools:
1. RAG Retriever - Retrieve relevant documents from ChromaDB collection
2. Web Search - Search the web using Tavily
3. Document Creator - Create downloadable Word documents

Plus utility functions for PDF processing and text cleaning.
"""

import os
import re
import unicodedata
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as WordDocument

# Load environment variables
load_dotenv()

def parse_pdf(filepath: str) -> List[Document]:
    """
    Parse a PDF file and extract text with metadata using LangChain's PyMuPDFLoader.
    
    Args:
        filepath (str): Path to the PDF file
        
    Returns:
        List[Document]: List of LangChain Document objects with page content and metadata
    """
    try:
        # Use LangChain's PyMuPDFLoader for PDF parsing
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        
        # Extract filename from filepath
        filename = os.path.basename(filepath)
        
        # Enhance metadata for each document
        for doc in documents:
            doc.metadata['filename'] = filename
            doc.metadata['text_format'] = 'text'
            doc.metadata['extraction_method'] = 'langchain_pymupdf'
            doc.metadata['has_tables'] = '|' in doc.page_content
            doc.metadata['char_count'] = len(doc.page_content)
            doc.metadata['word_count'] = len(doc.page_content.split())
            doc.metadata['line_count'] = len(doc.page_content.split('\n'))
        
        return documents
        
    except Exception as e:
        print(f"Error parsing PDF {filepath}: {str(e)}")
        return []


def clean_text(text: str) -> str:
    """
    Clean text for better RAG performance while preserving markdown structure.
    
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


def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 150) -> List[Document]:
    """
    Split LangChain Documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (List[Document]): List of LangChain Document objects
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[Document]: List of chunked Document objects with preserved metadata
    """
    if not documents:
        return []
    
    # Initialize LangChain's text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex = False,
    )
    
    # Split documents and preserve metadata
    chunked_docs = text_splitter.split_documents(documents)
    
    # Add chunk-specific metadata
    for i, doc in enumerate(chunked_docs):
        doc.metadata['chunk_number'] = i + 1
        doc.metadata['chunk_char_count'] = len(doc.page_content)
        doc.metadata['chunk_word_count'] = len(doc.page_content.split())
    
    return chunked_docs


def get_vectorstore(collection_name: str = "general_collection", persist_directory: str = "./chroma_db") -> Chroma:
    """
    Get or create a LangChain Chroma vectorstore.
    
    Args:
        collection_name (str): Name of the collection
        persist_directory (str): Directory to persist the vectorstore
        
    Returns:
        Chroma: LangChain Chroma vectorstore object
    """
    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
    
    vectorstore = Chroma(
        collection_name = collection_name,
        embedding_function = embeddings,
        persist_directory = persist_directory
    )
    
    return vectorstore


def process_and_store_pdf(filepath: str, collection_name: str = "general_collection", 
                          chunk_size: int = 500, chunk_overlap: int = 150) -> int:
    """
    Process a PDF file and store it in the Chroma vectorstore using LangChain.
    
    Args:
        filepath (str): Path to the PDF file
        collection_name (str): Name of the Chroma collection
        chunk_size (int): Size for text chunking
        chunk_overlap (int): Overlap for text chunking
        
    Returns:
        int: Number of chunks added to the vectorstore
    """
    # Parse PDF using LangChain
    documents = parse_pdf(filepath)
    
    if not documents:
        return 0
    
    # Clean text content
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    # Chunk documents using LangChain
    chunked_docs = chunk_documents(documents, chunk_size, chunk_overlap)
    
    # Get vectorstore and add documents
    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(chunked_docs)
    
    return len(chunked_docs)

@tool
def retrieve_documents(query: str, collection_name: str = "general_collection", top_k: int = 5) -> str:
    """
    Retrieve relevant documents from the RAG collection using semantic search.
    Use this tool when the user asks questions that might be answered by previously uploaded documents.
    
    Args:
        query: The search query to find relevant documents
        collection_name: Name of the ChromaDB collection (default: "general_collection")
        top_k: Number of top results to return (default: 5)
        
    Returns:
        str: Retrieved document contents with metadata
    """
    try:
        # Get vectorstore using LangChain
        vectorstore = get_vectorstore(collection_name)
        
        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(query, k = top_k)
        
        if not results:
            return "No relevant documents found in the collection."
        
        # Format results
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            content = doc.page_content
            
            result_text = f"--- Result {i} (Relevance Score: {score:.4f}) ---\n"
            result_text += f"Source: {metadata.get('filename', 'Unknown')}\n"
            result_text += f"Page: {metadata.get('page', 'N/A')}\n"
            result_text += f"Content:\n{content}\n"
            formatted_results.append(result_text)
        
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily to find current information.
    Use this tool when the user asks about current events, real-time information, or topics not in the uploaded documents.
    
    Args:
        query: The search query
        
    Returns:
        str: Search results with relevant information
    """
    # Initialize Tavily search
    tavily_search = TavilySearch(
                    max_results = 1, 
                    topic = "general", 
                    search_depth = "advanced",
                    include_answer = "advanced")
    try:
        # Use Tavily search for online content
        results = tavily_search.invoke(query)
        return f"Search results for '{query}':\n{results['answer']}"
    except Exception as e:
        return f"Error searching for '{query}': {str(e)}"

    
        


@tool
def create_document(content: str, filename: str = "document.docx", title: str = "Document") -> str:
    """
    Create a Word document (.docx) that the user can download.
    Use this tool when the user asks to create, generate, or save a document.
    
    Args:
        content: The text content to include in the document
        filename: Name of the output file (default: "document.docx")
        title: Title to display at the top of the document (default: "Document")
        
    Returns:
        str: Path to the created document or error message
    """
    try:
        # Ensure .docx extension
        if not filename.endswith(".docx"):
            filename += ".docx"
        
        # Create documents directory if it doesn"t exist
        docs_dir = Path("documents")
        docs_dir.mkdir(exist_ok = True)
        
        filepath = docs_dir / filename
        
        # Create a new Word document
        doc = WordDocument()
        
        # Add title
        if title:
            doc.add_heading(title, 0)
        
        # Add content (split by paragraphs)
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            if para.strip():
                doc.add_paragraph(para.strip())
        
        # Save the document
        doc.save(str(filepath))
        
        return f"✅ Document created successfully at: {filepath}\nThe file is ready for download."
    
    except Exception as e:
        return f"❌ Error creating document: {str(e)}"


# Helper function to get available tools
def get_all_tools():
    """Return all available tools for the agent."""
    return [retrieve_documents, web_search, create_document]