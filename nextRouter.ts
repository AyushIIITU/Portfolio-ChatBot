import { NextResponse } from 'next/server';
// import { graph } from '@/lib/chatbot-utils';

// Initialize chatbot when the API route is first loaded
// let isInitialized = false;
// import express from "express";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { HumanMessagePromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { ChatOllama } from "@langchain/ollama";
import { OpenAI } from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// const app = express();
// const port = 3000;

// app.use(express.json());

const embeddingModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});
const llm = new ChatOllama({
  model: "llama3.1:8b",
});

const vectorStore = new FaissStore(embeddingModel, {});
const cheerioLoader = new CheerioWebBaseLoader("https://portfolio-ayushiiitus-projects.vercel.app/");

const loadAndIndexDocs = async (): Promise<void> => {
  const docs = await cheerioLoader.load();
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const allSplits = await splitter.splitDocuments(docs);
  await vectorStore.addDocuments(allSplits);
};

await loadAndIndexDocs();

const humanTemplate = `You are a portfolio assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:`;

const promptTemplate = HumanMessagePromptTemplate.fromTemplate(humanTemplate);

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const retrieve = async (state: { question: string }): Promise<{ context: Document[] }> => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  return { context: retrievedDocs };
};

// const generate = async (state: typeof StateAnnotation.State): Promise<{ answer: string }> => {
//   const docsContent = state.context.map(doc => doc.pageContent).join("\n");
//   const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
//   const response = await llm.invoke(messages);
//   return { answer: response.content as string };
// };
const generate = async (state: typeof StateAnnotation.State): Promise<{ answer: string }> => {
  const docsContent = state.context.map(doc => doc.pageContent).join("\n");
  const prompt = `You are a portfolio assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.
  Use three sentences maximum and keep the answer concise.

  Question: ${state.question}
  Context: ${docsContent}
  Answer:`;

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: prompt }],
    temperature: 0,
  });

  return { answer: response.choices[0].message.content };
};

const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

export async function POST(req: Request) {
  try {
    // // Initialize chatbot if not already initialized
    // if (!isInitialized) {
    //   const success = await initializeChatbot();
    //   if (!success) {
    //     return NextResponse.json(
    //       { error: 'Failed to initialize chatbot' },
    //       { status: 500 }
    //     );
    //   }
    //   isInitialized = true;
    // }

    // Get the question from the request body
    const { question } = await req.json();

    if (!question) {
      return NextResponse.json(
        { error: 'Question is required' },
        { status: 400 }
      );
    }

    // Get response from chatbot
    const result = await graph.invoke({ question });
    console.log(result.answer);
    return NextResponse.json({ answer:result.answer });
  } catch (error) {
    console.error('Error in chat API:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 