from Supervised_Fine_Tuning import SFT_using_GPT_2
import streamlit as st

st.set_page_config (page_title = "RL-HumanFeedback Chatbot", page_icon = "🤖")

@st.cache_resource
def load_trained_model ():
    model = SFT_using_GPT_2 ()
    try:
        model.load_model ('RL')
        return model
    except Exception as e:
        st.error (f"Error loading model: {e}")
        return None

st.title ("🚀 RLHF Chatbot")
st.markdown ("This app applies a fine-tuned **GPT-2** policy, optimized via **PPO** and a **RoBERTa** reward model.")

if "model" not in st.session_state:
    with st.spinner ("Initializing weights and loading model..."):
        st.session_state.model = load_trained_model ()

prompt = st.chat_input ("Ask your RLHF model anything...")

if prompt:
    with st.chat_message ("user"):
        st.write (prompt)

    with st.chat_message ("role"):
        with st.status ("Model is thinking...", expanded = True) as status:
            st.write ("Processing tokens and generating response...")
            
            answer = st.session_state.model.predict (prompt, True)
            
            status.update (label = "Response generated!", state = "complete", expanded = False)
        
        st.write(answer)

with st.sidebar:
    st.header ("Model Stats")
    st.info ("Architecture: GPT-2 + PPO")
    if st.button ("Clear Chat"):
        st.rerun ()





# cd '/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install streamlit torch transformers pyspark
# streamlit run Inference.py


