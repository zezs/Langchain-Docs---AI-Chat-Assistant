import os
from dotenv import load_dotenv


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangchain
from langchain import hub #prompt community

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name=os.environ["INDEX_NAME"]

load_dotenv()

def run_llm(query: str):
    print("Retrieving...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(verbose=True, temperature=0)

    docsearch = Pinecone.from_documents(
        index_name=index_name, embeddings=embeddings
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        preturn_source_documents=True
    )
    
    return qa({"query": query})


if __name__=="__main__":
    result = run_llm(query="What is Langchain ?")
    # print(result["answer"])
    

    
