import streamlit as st
from chatbot_core import build_index, ask_chatbot, get_greeting, BOT_NAME
from knowledge_base import hospital_info

st.set_page_config(page_title=BOT_NAME, page_icon="🩺")

st.title("🩺 " + BOT_NAME)
st.caption("AI symptom assistant for " + hospital_info["name"] + ". Not a substitute for a real doctor.")


@st.cache_resource
def load_index():
    return build_index()


if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.followup_asked = False
    st.session_state.greeted = False

with st.spinner("Loading knowledge base..."):
    index = load_index()

if not st.session_state.greeted:
    st.session_state.history.append({"role": "assistant", "content": get_greeting()})
    st.session_state.greeted = True

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Describe your symptoms or ask a question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, score, specialty, followup_asked = ask_chatbot(
                query, index, st.session_state.history, st.session_state.followup_asked
            )
        st.write(answer)
        if score is not None:
            confidence_percent = round(score * 100, 1)
            st.caption("Matched specialty: " + str(specialty) + " | Confidence: " + str(confidence_percent) + "%")

    st.session_state.followup_asked = followup_asked
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})
