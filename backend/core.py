import os
from dotenv import load_dotenv
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub #prompt community

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()


def run_llm(query: str, chat_history: list[dict[str, Any]] = []):
    print("Retrieving...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(verbose=True, temperature=0)
    vectorstore = PineconeVectorStore( index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    #this prompt modifies the original question by combining it with chat history, gving us a better prompt that will help to reriever better result 
    #43. 5:00
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=vectorstore.as_retriever(), prompt=rephrase_prompt
    )
    
    retrival_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain)
    # retrival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    result = retrival_chain.invoke(input={"input": query, "chat_history":chat_history})
    
    return result


if __name__=="__main__":
    res = run_llm(query="What are agents?")

    

    
