import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
from langchain_groq import ChatGroq



import streamlit as st

# Set API Key
os.environ["GROQ_API_KEY"] = "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"

# Streamlit Framework
st.title('Celebrity Search Results')
input_text = st.text_input("Enter the name of a celebrity:")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about the celebrity {name}."
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# OpenAI LLM
llm = ChatGroq(model_name="mixtral-8x7b-32768")
print(llm) 

# Chains
chain = LLMChain(
    llm=llm, prompt=first_input_prompt, verbose=False, output_key='person', memory=person_memory
)

chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt, verbose=False, output_key='dob', memory=dob_memory
)

chain3 = LLMChain(
    llm=llm, prompt=third_input_prompt, verbose=False, output_key='description', memory=descr_memory
)

# Sequential Chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=False
)

# Streamlit Execution
if input_text:
    response = parent_chain({'name': input_text})
    st.write("**Celebrity Info:**", response['person'])
    st.write("**Date of Birth:**", response['dob'])
    st.write("**Major Events Around Birth Year:**", response['description'])

    with st.expander('Person Details History'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events History'): 
        st.info(descr_memory.buffer)
