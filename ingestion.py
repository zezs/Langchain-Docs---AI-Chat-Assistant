from dotenv import load_dotenv

import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader # Load documentation from ReadTheDocs for processing with LangChain
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    #loading
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest", encoding="UTF-8")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    #splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # print(documents)
    documents_str = json.dumps(documents)
    filename = "document_obj.json"
    with open(filename, "w") as file:
        file.write(documents_str)

    #updating metadat with new url
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    # embedding and finally storing in vectorDB
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name="langchain-docs-ai")
    print("**** Loading to vectorstore done ****")

if __name__ == "__main__":
    ingest_docs()