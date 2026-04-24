import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["OPENAI_API_KEY"] = ""

loader = TextLoader("", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],
    chunk_size=300,
    chunk_overlap=0,
)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
 """

請回答，但只能根據以下提供的資料來回答，不能使用其他資料來源。

參考資料：
{context}

使用者描述：
{question}

回答："""
)

llm = ChatOpenAI(
    model="",
    temperature=0.1,
    base_url="https://openrouter.ai/api/v1",
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    while True:
        question = input("請描述問題： ")
        if question.strip().lower() == "quit":
            break
        answer = qa_chain.invoke(question)
        print(f"\n{answer}\n")