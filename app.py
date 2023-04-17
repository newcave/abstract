import requests
import json
from bs4 import BeautifulSoup
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator
import streamlit as st

# Set the endpoint URL and parameters
url = "https://api.elsevier.com/content/search/sciencedirect"
doi = "10.1016/j.jtbi.2021.110684"
query = f"doi('{doi}')"
api_key = "c551fb6c03984e2a1bb4afa68fc94534"

# Set up the headers and authentication
headers = {
    "Accept": "application/json",
    "X-ELS-APIKey": api_key
}




# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')


# def extract_abstract(url):
#     try:
#         webpage = requests.get(url)
#         soup = BeautifulSoup(webpage.text, "html.parser")
#         abstract = soup.find(class_='html-p').text
#         return abstract
#     except requests.exceptions.RequestException:
#         st.error("Invalid URL. Please refresh the app and try again.")
#         return None

# def extract_abstract(url):
#     webpage = requests.get(url)
#     soup = BeautifulSoup(webpage.text, "html.parser")
#     abstract = soup.find(class_='html-p').text
#     return abstract

def summarize_abstract(abstract):
    # Encode the input text
    input_ids = tokenizer.encode(abstract, return_tensors='pt', max_length=512)

    # Generate the summary
    summary_ids = model.generate(input_ids,
                                 min_length=100,
                                 max_length=300,
                                 length_penalty=2.0,
                                 num_beams=4,
                                 early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def translate_summary(summary):
    translator = Translator()
    translated = translator.translate(summary, dest='ko')
    return translated.text

st.title("MDPI Paper Summarizer made by Dr. KWJOO, decorated by newcave")

url = st.text_input("Enter MDPI URL")

if url:
    abstract = extract_abstract(url)
    st.subheader("Abstract")
    st.write(abstract)
    
    summary = summarize_abstract(abstract)
    st.subheader("Summary")
    st.write(summary)
    
    translated = translate_summary(summary)
    st.subheader("Translated Summary")
    st.write(translated)
