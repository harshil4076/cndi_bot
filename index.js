//1. Import necessary packages
import { GithubRepoLoader } from 'langchain/document_loaders/web/github'
import { OpenAI } from "langchain/llms/openai";
import {RetrievalQAChain} from 'langchain/chains'
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import fs from 'fs'
import * as dotenv from 'dotenv'

dotenv.config()
//Use Github to query cndi
const question = "how to create kubernetes cluster?";
//name of vector store.
//Will create a folder named cndi_store.index locally
//May take some time to create the store for the first time
const VECTOR_STORE_PATH = "cndi_store.index"

const runCndiChatBot = async () => {
    //initialize the model
    const model = new OpenAI({modelName: "gpt-3.5-turbo"})

    let vectorStore;
    //check if vector store exist locally
    if(fs.existsSync(VECTOR_STORE_PATH)){

      console.log('Vector Exist..')
      vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings())

    }else {
      //load cndi files
        const cndi_loader = new GithubRepoLoader(
            "https://github.com/polyseam/cndi",
            {branch: "main", recursive:false, unknown: "warn", accessToken:process.env.GITHUB_ACCESS_TOKEN}
        )
        //convert it to text
        const docs = await cndi_loader.loadAndSplit();
        // create vector store
        vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

         await vectorStore.save(VECTOR_STORE_PATH)

    }
    //chain embeddings and openai
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    const res = await chain.call({
      query: question,
    })

    console.log({res})
}

runCndiChatBot()