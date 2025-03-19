import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

import { FaissStore } from "@langchain/community/vectorstores/faiss";

import { HumanMessagePromptTemplate } from "@langchain/core/prompts";
import { ChatOllama } from "@langchain/ollama";
const embeddingModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});
const llm = new ChatOllama({
  model: "llama3.1:8b",  // Default value.
});

// const result = await model.invoke(["human", "Hello, how are you?"]);
// console.log(result);  // Output: ["ollama", "I'm fine, thank you."]16149645600 21455957700

const vectorStore = new FaissStore(embeddingModel, {});

// Load and chunk contents of blog
// const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://portfolio-ayushiiitus-projects.vercel.app/"
);

const docs = await cheerioLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000, chunkOverlap: 200
});
const allSplits = await splitter.splitDocuments(docs);


// Index chunks
await vectorStore.addDocuments(allSplits)

const humanTemplate = `You are a portfolio assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:`;

const promptTemplate = HumanMessagePromptTemplate.fromTemplate(humanTemplate);

// Define state for application
const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

// Define application steps
const retrieve = async (state: typeof InputStateAnnotation.State) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question)
  return { context: retrievedDocs };
};


const generate = async (state: typeof StateAnnotation.State) => {
  const docsContent = state.context.map(doc => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};


// Compile application and test
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();
  let inputs = { question: "What are the techstack?" };

  const result = await graph.invoke(inputs);
  // for (const stream_output of graph.astream(inputs)) {
  //     console.log(stream_output);
  // }
  console.log(result.answer);