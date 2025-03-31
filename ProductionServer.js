import express from "express";
import fs from "fs";
import path from "path";
import "cheerio";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAI } from "openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const port = process.env.PORT || 3100;

app.use(express.json());

const openai = new OpenAI({
  baseURL: "https://api.studio.nebius.com/v1/",
  apiKey: process.env.NEBIUS_API_KEY,
});

const embeddingModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

const faissIndexPath = path.join(process.cwd(), "faiss_index"); // Directory for FAISS index
let vectorStore;

const loadAndIndexDocs = async () => {
    if (fs.existsSync(path.join(faissIndexPath, "faiss.index")) &&
        fs.existsSync(path.join(faissIndexPath, "docstore.json"))) {
      console.log("Loading FAISS index from saved files...");
  
      // Correct way to load FAISS
      vectorStore = await FaissStore.load(
        faissIndexPath, 
        embeddingModel, 
        { docstorePath: path.join(faissIndexPath, "docstore.json") } // Explicitly specify docstore
      );
  
      console.log("FAISS index successfully loaded.");
    } else {
      console.log("No FAISS index found. Scraping website and creating index...");
  
      // Scrape and process data
      const cheerioLoader = new PuppeteerWebBaseLoader(
        "https://portfolio-ayushiiitus-projects.vercel.app/"
      );
      const docs = await cheerioLoader.load();
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      const allSplits = await splitter.splitDocuments(docs);
  
      vectorStore = new FaissStore(embeddingModel, {});
      await vectorStore.addDocuments(allSplits);
  
      // Ensure the directory exists
      if (!fs.existsSync(faissIndexPath)) {
        fs.mkdirSync(faissIndexPath, { recursive: true });
      }
  
      // Save FAISS index and metadata
      await vectorStore.save(faissIndexPath);
      
      console.log("FAISS index successfully created and saved.");
    }
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

const generate = async (state) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const prompt = `You are a portfolio assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.

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

app.post("/ask", async (req, res) => {
  const { question } = req.body;
  if (!question) {
    res.status(400).json({ error: "Question is required" });
    return;
  }
  try {
    const result = await graph.invoke({ question });
    res.json({ answer: result.answer });
  } catch (error) {
    res
      .status(500)
      .json({ error: "Internal server error", details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
