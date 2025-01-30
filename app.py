import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

import os
load_dotenv()

##load groq api
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')



st.set_page_config(page_title="Text to Math problem solver assistant")
st.title("Text to Math Problem Solver Using Google Gemma")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not groq_api_key:
    st.info("please add your groq api to continue")
    st.stop()

llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it")

#initializing the tools
wiki_wrapper=WikipediaAPIWrapper()
wiki_tool=Tool(
    name="wikipedia",
    func=wiki_wrapper.run,
    description="Search Wikipedia for information to solve math problems",
)

##initialize the math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(name='calculation',
                func=math_chain.run,
                description="a tool for answering maths related questions.only input mathematical expression needs to be provided")

prompt="""

your an agent tasked for solving user 
mathematical questions.Logically arrive at a 
solution and provide the detailed explanation 
for it and display 
it step by step for the question below
question:{question}
answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

##COMBINE ALL THE TOOLS INTO CHAIN
chain=LLMChain(llm=llm,prompt=prompt_template)
reasoning_tool=Tool(name="reasoning tool",
                    func=chain.run,
                    description="A tool that can answer logic based and reasoning questions")

agent=initialize_agent(tools=[wiki_tool,calculator,reasoning_tool],
                       llm=llm,
                       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=False,
                       handling_parsing_errors=True
                       )


if "messages" not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant','content':'HI!! I am a math chatbot who can answer all your math problems'}
    ]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])  




question=st.chat_input(placeholder="enter your question:")

if question:
    with st.spinner("Generating response...."):
            
        st.session_state.messages.append({'role':"user","content":question})
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=agent.run(question,callbacks=[st_cb]) 
            st.session_state.messages.append({'role':"assistant","content":response}) 
            st.write(response)

else:
        st.warning('please enter the question')        




