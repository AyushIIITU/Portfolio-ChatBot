import { ChatOllama } from "@langchain/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { HumanMessagePromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import link from "./Link.js";
import fs from "fs/promises";
import path from "path";

const embeddingModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});
const llm = new ChatOllama({
  model: "llama3.1:8b",
});

const urls = link.filter(url => url && typeof url === "string");
console.log("URLs from Link.js (filtered):", urls);
if (urls.length === 0) {
  throw new Error("No valid URLs found in Link.js");
}

const loadDocuments = async () => {
  const loaders = urls.map((url) => new PuppeteerWebBaseLoader(url, {
    launchOptions: { headless: "new" },
    gotoOptions: { waitUntil: "networkidle2", timeout: 300000 }, // 5-minute timeout
  }));
  const docsArrays = await Promise.all(
    loaders.map(async (loader) => {
      try {
        const docs = await loader.load();
        console.log(`Loaded ${docs.length} documents from ${loader.url}`);
        return docs;
      } catch (error) {
        console.error(`Error loading ${loader.url}:`, error.message);
        return [];
      }
    })
  );
  const documents = docsArrays.flat();
  console.log("Total loaded documents:", documents.length);
  return documents;
};

const initializeVectorStore = async () => {
  const documents = await loadDocuments();
  if (documents.length === 0) {
    throw new Error("No documents loaded. Check URLs and network access.");
  }
  console.log("Sample document content (first 100 chars):", 
    documents[0].pageContent.substring(0, 100) || "No content");
  console.log("First 5 documents preview:");
  documents.slice(0, 5).forEach((doc, i) => {
    console.log(`Doc ${i + 1} (length: ${doc.pageContent.length}):`, 
      doc.pageContent.substring(0, 50) || "[empty]");
  });

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });
  if (documents.length > 0) {
    const testSplit = await textSplitter.splitText(documents[0].pageContent);
    console.log("Test split of first document:", testSplit);
  }
  const allSplits = await textSplitter.splitDocuments(documents);
  console.log("Number of document splits:", allSplits.length);
  if (allSplits.length === 0) {
    throw new Error("No document splits found. Check document content and splitter settings.");
  }

  const vectorStore = await FaissStore.fromDocuments(allSplits, embeddingModel);
  console.log("Vector store initialized successfully.");
  const indexPath = path.resolve("./faiss_index");
  await vectorStore.save(indexPath);
  console.log("Vector store saved to disk at:", indexPath);
  return vectorStore;
};

(async () => {
  try {
    const vectorStore = await initializeVectorStore();
    const loadedVectorStore = await FaissStore.load("./faiss_index", embeddingModel);
    console.log("Vector store loaded from disk successfully.");

    const humanTemplate = `You are an assistant for an institute's website. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Question: {question}
Context: {context}
Answer:`;
    const promptTemplate = HumanMessagePromptTemplate.fromTemplate(humanTemplate);

    const StateAnnotation = Annotation.Root({
      question: Annotation,
      context: Annotation,
      answer: Annotation,
    });

    const retrieve = async (state) => {
      const retrievedDocs = await loadedVectorStore.similaritySearch(state.question, 4);
      return { context: retrievedDocs };
    };

    const generate = async (state) => {
      const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
      const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
      const response = await llm.invoke(messages);
      return { answer: response.content };
    };

    const graph = new StateGraph(StateAnnotation)
      .addNode("retrieve", retrieve)
      .addNode("generate", generate)
      .addEdge("__start__", "retrieve")
      .addEdge("retrieve", "generate")
      .addEdge("generate", "__end__")
      .compile();

    const inputs = { question: "Give placement Statistics?" };
    const result = await graph.invoke(inputs);
    console.log("Answer:", result.answer);
  } catch (error) {
    console.error("Error:", error);
  }
})();