from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


st.header("Langchain Docs 🦜🔗 - AI Chat Assistant")

# custom func
def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# init session state varibles which hold data persistant until the session is over
# implementing chat history
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


prompt = st.text_input("Prompt", placeholder="Enter your prompt here...") or st.button("Submit")

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"]) #chat_history-> list of tuples, [tuple (first ele->role, second element->content)]
       
        # gathering sources/ set to avoid duplicates
        # sources = set( 
        #     [doc.metadata["source"] for doc in generated_response["context"]]
        #     )
        sources = set(doc.metadata["source"] for doc in generated_response["context"])
        
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        # storing query and response
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

        #storing chat history
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))


# displaying previosuly entered prompts and generated answers
if st.session_state["chat_answers_history"]:
    for genearted_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(user_query, is_user=True) # message func is from streamlit chat package 
        message(genearted_response)
        
