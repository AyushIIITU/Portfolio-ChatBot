import { ChatOllama } from "@langchain/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { HumanMessagePromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

// Define embedding model
const embeddingModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

// Define LLM
const llm = new ChatOllama({
  model: "llama3.1:8b",
});

// Async function to run the process
(async () => {
  // Initialize vector store
  const vectorStore = new FaissStore(embeddingModel, {});

  // Load website data
  const cheerioLoader = new CheerioWebBaseLoader("https://portfolio-ayushiiitus-projects.vercel.app/");
  const documents: Document[] = await cheerioLoader.load();

  // Split text into smaller chunks
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await textSplitter.splitDocuments(documents);

  // Index chunks in the vector store
  await vectorStore.addDocuments(allSplits);

  // Define prompt for question-answering
  const humanTemplate = `You are a portfolio assistant for question-answering tasks. 
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.
  Use three sentences maximum and keep the answer concise.

  Question: {question}
  Context: {context}
  Answer:`;

  const promptTemplate = HumanMessagePromptTemplate.fromTemplate(humanTemplate);

  // Define state types
  interface InputState {
    question: string;
  }

  interface State extends InputState {
    context: Document[];
    answer: string;
  }

  // Define retrieval step
  const retrieve = async (state: InputState): Promise<Partial<State>> => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question, 3); // Limit results to 3
    return { context: retrievedDocs };
  };

  // Define generation step
  const generate = async (state: State): Promise<Partial<State>> => {
    const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
    const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
    const response = await llm.invoke(messages); // Pass the message directly
    return { answer: Array.isArray(response.content) ? response.content.join(" ") : response.content || "" };
  };

  // Compile application state graph
  const graph = new StateGraph()
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

  // Run pipeline
  const inputs: InputState = { question: "What are the techstack?" };
  const result = await graph.invoke(inputs);

  console.log("Answer:", result.answer);
})();
