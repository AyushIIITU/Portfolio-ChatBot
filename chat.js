import { ChatOllama } from "@langchain/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
// import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate, HumanMessagePromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { PromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HumanMessage } from "@langchain/core/messages";
const embeddingModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});
const llm = new ChatOllama({
  model: "llama3.1:8b",  // Default value.
});

// const result = await model.invoke(["human", "Hello, how are you?"]);
// console.log(result);  // Output: ["ollama", "I'm fine, thank you."]16149645600 21455957700

const vectorStore = new FaissStore(embeddingModel, {});
const cheerioLoader = new CheerioWebBaseLoader(
  "https://portfolio-ayushiiitus-projects.vercel.app/"
);
const document = await cheerioLoader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000, chunkOverlap: 200
});
const allSplits = await textSplitter.splitDocuments(document);
// Index chunks
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

// console.log(promptTemplate);
// Define state for application
const InputStateAnnotation = Annotation.Root({
  question: Annotation,
});
const StateAnnotation = Annotation.Root({
  question: Annotation,
  context: Annotation,
  answer: Annotation,
});
// Define application steps
const retrieve = async (state) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question)
  return { context: retrievedDocs };
};
const generate = async (state) => {
  const docsContent = state.context.map(doc => doc.pageContent).join("\n");
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
let inputs = { question: "What are the techstack?" };

const result = await graph.invoke(inputs);
// for (const stream_output of graph.astream(inputs)) {
//     console.log(stream_output);
// }
console.log(result.answer);