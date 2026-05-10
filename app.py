import streamlit as st
from query import ask

st.title("📚 Course AI Tutor")
st.caption("Ask anything — I'll answer from your course content first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about the course..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})