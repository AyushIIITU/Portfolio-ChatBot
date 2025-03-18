# Chatbot Backend for Portfolio Q&A

## Overview
This project implements a chatbot backend that scrapes data from a personal portfolio and utilizes it for question-answering. The chatbot is built using LangChain, FAISS for vector storage, and Ollama LLaMA 3 for natural language processing. It enables users to ask questions related to the portfolio, retrieving contextually relevant answers.

## Features
- **Web Scraping**: Extracts content from a portfolio website using Cheerio.
- **Text Splitting**: Uses RecursiveCharacterTextSplitter to segment scraped content into manageable chunks.
- **Vector Embeddings**: Converts textual data into vector embeddings using `HuggingFaceTransformersEmbeddings`.
- **FAISS Vector Store**: Stores document embeddings for efficient similarity search.
- **StateGraph Execution**: Implements a structured retrieval and generation pipeline for answering user queries.
- **Chatbot Integration**: Utilizes the LLaMA 3 model via Ollama to generate responses based on retrieved context.

## Tech Stack
- **Node.js**
- **LangChain**
- **FAISS (Facebook AI Similarity Search)**
- **Hugging Face Transformers**
- **Ollama (LLaMA 3.1 8B model)**
- **Cheerio (Web Scraping)**

## Setup & Installation
### Prerequisites
- Node.js (v16+)
- Install dependencies using npm or yarn

```sh
npm install
```

### Configuration
Ensure the portfolio URL is correctly set in the `CheerioWebBaseLoader`.

### Running the Project
Execute the script to start the chatbot backend:

```sh
node index.js
```

## Workflow
1. **Web Scraping**: The portfolio content is scraped using `CheerioWebBaseLoader`.
2. **Text Processing**: Text is split into chunks to optimize vector embeddings.
3. **Embedding & Storage**: The chunks are transformed into vector representations and stored in FAISS.
4. **Retrieval & Response Generation**:
   - Given a user question, relevant context is retrieved using similarity search.
   - A structured prompt is created using `ChatPromptTemplate`.
   - The chatbot generates a concise response using the LLaMA model.

## Example Usage
```javascript
let inputs = { question: "What are the techstack?" };
const result = await graph.invoke(inputs);
console.log(result.answer);
```
```javascript
The tech stack used includes:

* React js, Node js, Bootstrap, MongoDB (Campus Management System)
* React js, Tailwind CSS, Node processing, WebSocket integration, YOLO, and MongoDB (RoadSense)
* Python, Ollama, LangChain, Flutter, and SQLlite (GeinAi AI Content Generation & MCQ Platform)
* JavaScript, Node.js, CSS, HTML, and MySQL (BlindKart E-commerce Website)
```
```javascript
let inputs = { question: "What are the a+b" };
const result = await graph.invoke(inputs);
console.log(result.answer);
```
```javascript
I don't know what the equation "a+b" refers to. The provided context appears to be a portfolio or resume for a software developer with 
various projects, but there is no specific information about mathematical equations.
```
## Future Improvements
- Implement API endpoints for chatbot interaction.
- Enhance scraping capabilities for more dynamic data.
- Optimize response generation with prompt engineering.
- Deploy as a backend service with REST or WebSocket support.

## License
This project is licensed under the MIT License.

