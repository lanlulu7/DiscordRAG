from discord import Intents
from discord.ext.commands import Bot
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import shutil
import os

f = open('key.txt', 'r')
token = f.readline().replace('\n', '')
key = f.readline().replace('\n', '')
f.close()


url = ''

db_name = 'db'
i = 0

intents = Intents.default()
intents.message_content = True

bot = Bot(command_prefix='!', intents=intents)


@bot.command()
async def hello(context):
    await context.send('Hello!')


@bot.command()
async def upload(context):
    print(context.message.attachments[0].url)
    global url
    url = context.message.attachments[0].url
    await context.send('上傳完成')


@bot.command()
async def q(context, query):
    global i, db_name
    if os.path.exists(db_name):
        shutil.rmtree(db_name)
        i += 1
        db_name = f'db_{i}'
    message = await context.send('請稍後')
    #load PDF
    loader = PyMuPDFLoader(url)
    PDF_data = loader.load()

    #Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(PDF_data)

    #Load Embedding model
    embedding = OpenAIEmbeddings(
        api_key=key, base_url='https://api.chatanywhere.tech/v1',
    model='text-embedding-3-large')
    #Load Embedding to VectorDB
    persist_directory = db_name # 'db'
    vectordb = Chroma.from_documents(documents=all_splits,
                                     embedding=embedding,
                                     persist_directory=persist_directory)

    #load gpt-3.5-turbo
    llm = ChatOpenAI(
        api_key=key, base_url='https://api.chatanywhere.tech/v1',
    temperature=0)

    #LLM combine with database
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever)

    #chat
    answer = qa.invoke(query)
    print(answer)
    await message.edit(content=f"GPT-3.5-turbo:\n{answer['result']}")
    #error
@bot.event
async def on_command_error(context, error):
    await context.send(error)

bot.run(token)