import re
from numpy import size
import streamlit as st
import io
import requests
import openai
from langchain.chat_models import ChatOpenAI
import os
from qdrant_client import QdrantClient
from langchain.embeddings.openai import OpenAIEmbeddings
import time
from qdrant_client.http import models
import tiktoken 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, OpenAI, LLMChain

#create requirements file 


# Get AWS credentials from Streamlit secrets
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]

#S3 configuration
s3_endpoint_url = f'https://s3.{AWS_DEFAULT_REGION}.amazonaws.com'
s3_bucket_name = 'your-bucket-name'  # Replace with your S3 bucket name

#qdrant configuration
QDRANT_SECRET_API_KEY = st.secrets["QDRANT_SECRET_API_KEY"]
qdrant_client = QdrantClient(
    url="https://9b8c4946-25cb-4ed6-9953-9ebb6b7406d3.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key=QDRANT_SECRET_API_KEY )

#OpenAI configuration
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
embedding_llm = OpenAIEmbeddings(model ="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', 
             temperature=0,
             openai_api_key=OPENAI_API_KEY)

#global variables for the application
retrieved_vectors = None
openai_result = None
evidence_dictionary = None

#state variable
st.session_state['evidence_dictionary_list'] = []

def callOpenAI(userQuery):

    #LANGCHAIN CONFIG
    template = """You are to act as Aristotle the philosopher. The user will ask you questions and I want you to answer based on what Aristotle would do or say. I've attached quotes from Aristotle himself to help you answer the user's questions as his persona. When answering my questions use the contexual evidence only and do not derive information outside of the contextual evidence. Do not contrive or fabricate answers."
    Evidence:{evidence}
    Question:{question}
    Answer: Think step by step."""
       
    prompt = PromptTemplate(template=template, input_variables=["question","evidence"])

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    global openai_result
    openai_result = llm_chain.run({"evidence" : retrieved_vectors, "question" : userQuery}) 
    
def embedQuestion(embedding): 
    global retrieved_vectors
    global evidence_dictionary
    #sanitize for length of query
    word_count = len(embedding.split())
    if word_count > 100:
        return;
    #embed question    
    while True:
            try:
                res = openai.Embedding.create(input=embedding, engine='text-embedding-ada-002')
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Waiting for 1 minute before retrying...")
                time.sleep(60)  
    vectors = qdrant_client.search(collection_name="aristotle", query_vector=res['data'][0]['embedding'], limit=4) 
    
    retrieved_vectors = vectors
    
    for point in retrieved_vectors:      
        evidence_dictionary = {"evidence_text" : point.payload['text'], "evidence_source" : point.payload['source'] }
        st.session_state.evidence_dictionary_list.append(evidence_dictionary)

     
with st.form("my_form"):
   st.write("Query the entity...")
   userQuery = st.text_area("")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
      embedQuestion(userQuery)
      callOpenAI(userQuery) 

st.write(openai_result)
with st.expander("See explanation"):
    for dictionary in st.session_state.evidence_dictionary_list:
        st.write('"' + dictionary["evidence_text"] + '"')
        st.write("Source: " + dictionary["evidence_source"])
        st.write("\n\n")
    
uploaded_file = st.file_uploader("Upload a file")


if uploaded_file:
   st.write("Filename: ", uploaded_file.name)
   
   
