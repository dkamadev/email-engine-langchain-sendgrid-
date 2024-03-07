import os

# import flask dependencies
from flask import Flask, request
from flask_mail import Mail, Message

# import langchain dependencies
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# load env
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.sendgrid.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'apikey'
app.config['MAIL_PASSWORD'] = os.environ.get('SENDGRID_API_KEY')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
mail = Mail(app)

db = None
llm = OpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'))

def scrape_page(web_url):
    loader = AsyncChromiumLoader([web_url])
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    documents = bs_transformer.transform_documents(html, tags_to_extract=["span", "p"])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

def embed(docs):
    global db
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return

def generate_email(query):
    docs = db.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    content =  chain.run(input_documents=docs, question=query)
    return content

@app.route('/knowledge', methods=['POST'])
def knowledge():
    url = request.json['web_url']
    docs = scrape_page(url)
    embed(docs)                
    return "Knowledge base generated"

@app.route('/emails', methods=['POST'])
def emails():
    query = request.json['query']
    email = request.json['email']
    title = request.json['title']
    msg = Message(title, recipients=[email])
    msg.body = generate_email(query)
    mail.send(msg)
    return "Email Sent"

if __name__ == '__main__':
    app.run(debug=True)
    
