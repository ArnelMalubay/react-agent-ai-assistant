# Job Application Assistant

An AI-powered assistant built with LangGraph and LangChain that helps with job applications by reading resumes, searching for job postings, and creating cover letters.

## Features

- **Resume Reader**: Extract text content from PDF resumes using PyMuPDF4LLM
- **Job Search**: Search for job postings and extract content from URLs using Tavily
- **Document Creator**: Generate Word documents for cover letters and other job application materials
- **Conversational AI**: Interactive chat interface for seamless job application assistance

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
GROQ_API_KEY=your-groq-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

### 3. Get API Keys

- **Groq API Key**: Get your API key from [Groq Console](https://console.groq.com/keys)
- **Tavily API Key**: Get your API key from [Tavily](https://tavily.com/)

## Usage

Run the application:

```bash
python app/app.py
```

### Example Interactions

1. **Reading a Resume**:
   ```
   You: Please read my resume at /path/to/resume.pdf
   ```

2. **Searching for Jobs**:
   ```
   You: Search for software engineer jobs in San Francisco
   You: Get details from this job posting: https://example.com/job-posting
   ```

3. **Creating Cover Letters**:
   ```
   You: Create a cover letter for a software engineer position
   ```

## Architecture

The application uses LangGraph to create a workflow with the following structure:

```
START → agent → conditional edge → [tools | END]
tools → agent
```

- **Agent Node**: Processes user messages and decides on actions
- **Tool Node**: Executes the appropriate tools (resume reader, job search, document creator)
- **Conditional Routing**: Determines whether to use tools or end the conversation

## Tools

### 1. Resume Reader (`resume_reader`)
- **Input**: File path to PDF resume
- **Output**: Extracted text content
- **Technology**: PyMuPDF4LLM

### 2. Job Search (`job_search`)
- **Input**: Search query or job posting URL
- **Output**: Search results or extracted content
- **Technology**: Tavily Search API

### 3. Document Creator (`document_creator`)
- **Input**: Text content and filename
- **Output**: Word document (.docx)
- **Technology**: python-docx

## File Structure

```
job-application-assistant/
├── app/
│   └── app.py              # Main application
├── documents/              # Generated documents (auto-created)
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## License

MIT License - see LICENSE file for details.