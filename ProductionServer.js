import express from "express";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
// import { Document } from "@langchain/core/documents";
// import { HumanMessagePromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
// import { ChatOllama } from "@langchain/ollama";
import { OpenAI } from "openai";
import dotenv from "dotenv";
dotenv.config();

const app = express();
const port = 3100;

app.use(express.json());
const openai = new OpenAI({
    baseURL: 'https://api.studio.nebius.com/v1/',
    apiKey: process.env.NEBIUS_API_KEY,
  });

const embeddingModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});
// const llm = new ChatOllama({
//   model: "llama3.1:8b",
// });

const vectorStore = new FaissStore(embeddingModel, {});
const cheerioLoader = new CheerioWebBaseLoader("https://portfolio-ayushiiitus-projects.vercel.app/");

const loadAndIndexDocs = async ()=> {
  const docs = await cheerioLoader.load();
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const allSplits = await splitter.splitDocuments(docs);
  await vectorStore.addDocuments(allSplits);
};

await loadAndIndexDocs();



const StateAnnotation = Annotation.Root({
  question: Annotation,
  context: Annotation,
  answer: Annotation,
});

const retrieve = async (state) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  return { context: retrievedDocs };
};

const generate = async (state)=> {
  const docsContent = state.context.map(doc => doc.pageContent).join("\n");
  const prompt = `You are a portfolio assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.
  Use three sentences maximum and keep the answer concise.

  Question: ${state.question}
  Context: ${docsContent}
  Answer:`;

  const response = await openai.chat.completions.create({
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages: [{ role: "user", content: prompt }],
    temperature: 0,
    top_p: 0.9,
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

app.post("/ask", async (req, res)=> {
  const { question } = req.body;
  if (!question) {
    res.status(400).json({ error: "Question is required" });
    return;
  }
  try {
    const result = await graph.invoke({ question });
    res.json({ answer: result.answer });
  } catch (error) {
    res.status(500).json({ error: "Internal server error", details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
