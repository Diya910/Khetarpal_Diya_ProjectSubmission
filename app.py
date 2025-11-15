import streamlit as st
from bedrock_utils import (
    query_knowledge_base,
    generate_response,
    valid_prompt,
    extract_text
)

st.title("Bedrock Knowledge Base Chatbot")

st.sidebar.header("Configuration")
model_id = st.sidebar.selectbox(
    "Choose LLM Model",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ]
)

kb_id = st.sidebar.text_input("Knowledge Base ID", "your-knowledge-base-id")
temperature = st.sidebar.select_slider("Temperature", [i / 10 for i in range(0, 11)], 1)
top_p = st.sidebar.select_slider("Top_P", [i / 1000 for i in range(0, 1001)], 1)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about heavy machinery..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if valid_prompt(prompt, model_id):

        kb_results = query_knowledge_base(prompt, kb_id)
        context = "\n".join([
            extract_text(result)
            for result in kb_results
            if extract_text(result)
        ])

        full_prompt = f"Context:\n{context}\n\nUser Query: {prompt}\n\nAnswer clearly and concisely."

        response = generate_response(full_prompt, model_id, temperature, top_p)

    else:
        response = "I am unable to answer this. Please ask something related to heavy machinery."

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
