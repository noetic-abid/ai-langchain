import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import streamlit as st

# What is PromptTemplate ?

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

st.title('Celebrity Search')
text_input = st.text_input('Search the topic you want ..')

## Open AI Model initialization
llm = OpenAI(temperature=0.8)

# Define a new Prompt
my_first_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

chain = LLMChain(llm=llm , prompt=my_first_prompt,verbose=True)


if text_input:
    st.write(chain.run(text_input))
    
    