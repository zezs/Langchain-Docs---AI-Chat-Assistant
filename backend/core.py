import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub #prompt community

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


def run_llm(query: str):
    print("Retrieving...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(verbose=True, temperature=0)
    vectorstore = PineconeVectorStore( index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    # retrival_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(),
    #     prompt=retrieval_qa_chat_prompt
    # )

    result = retrival_chain.invoke(input={"input": query})
    
    return result


if __name__=="__main__":
    result = run_llm(query="What is Langchain Chain?")
    print(result["answer"])
    

    
