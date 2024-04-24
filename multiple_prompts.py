import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st


# New topics in this POC
## What is SequentialChain ? Use to Run Multiple Chains and each chain has different Prompt
## What is SimpleSequentialChain ? Always show latest output value and that is why we prefer SequentialChain over it.
## What is LLMChain ? To execute any propmt we use LLMChain
# Why we use ConversationBufferMemory ? To save the full chain in some store

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

st.title('Celebrity Search')
text_input = st.text_input('Search the topic you want ..')

person_memory = ConversationBufferMemory(input_key='name',memory_key='person_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='dob_history')
events_memory = ConversationBufferMemory(input_key='dob',memory_key='events_history')

## Open AI Model initialization
llm = OpenAI(temperature=0.8)

# Define first Prompt
prompt1 = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)
chain1 = LLMChain(llm=llm , prompt=prompt1,verbose=True,output_key='person',memory=person_memory)

# Define 2nd Prompt
prompt2 = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born ?"
)
chain2 = LLMChain(llm=llm , prompt=prompt2,verbose=True,output_key='dob',memory=dob_memory)

# Define 3rd Prompt
prompt3 = PromptTemplate(
    input_variables=['dob'],
    template="Please tell top five events happened around {dob} date "
)
chain3 = LLMChain(llm=llm , prompt=prompt3,verbose=True , output_key='events',memory=events_memory)

parent_chains = SequentialChain(
    chains=[chain1,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','dob','events'],verbose=True)


# Run All the chains
if text_input:
    st.write(parent_chains({'name':text_input}))
    with st.expander('Who is ? '):st.info(person_memory.buffer)
    with st.expander('Date of Birth ?'):st.info(dob_memory.buffer)
    with st.expander('Major Events around the DOB ? '):st.info(events_memory.buffer)


    
    