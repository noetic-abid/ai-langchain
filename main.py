import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI

import streamlit as st

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

st.title('LangChain Demo with OpenAI')
text_input = st.text_input('Serch the topic you want ..')

## Open AI Model initialization
llm = OpenAI(temperature=0.8)

if text_input:
    st.write(llm(text_input))
    
    