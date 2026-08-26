# Generated from the completed Break Through Tech notebook.

import os
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

client = OpenAI()

documents = TextLoader("data/marlowe_knowledge_base.txt").load()
chunk_size = 500
chunk_overlap = 75
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(documents)

average_length = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
print("Total number of chunks:", len(chunks))
print("Average chunk length:", average_length)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

rag_instruction = """
You are a customer support assistant for Marlowe & Finch.
Answer the customer's question using only the information provided in the context below.
Do not make up or assume information that is not included in the context.
If the answer cannot be found in the context, say that you cannot find that information in the knowledge base.
Keep your answer clear, friendly, professional, and concise.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(rag_instruction)
llm = ChatOpenAI(model="gpt-4o")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

baseline_queries = [
    "is the tent waterproof",
    "can i return a tent i used on a weekend trip",
    "do you ship to australia"
]

baseline_answers = {q: rag_chain.invoke(q) for q in baseline_queries}

rewrite_prompt_template = """
Rewrite the following customer query into a more complete and descriptive search query while preserving the customer's original intent.
Return exactly one rewritten query. Do not provide multiple options, explanations, or additional text.

Customer query:
{short_query}

Rewritten query:
"""

rewritten_queries = {}
for q in baseline_queries:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": rewrite_prompt_template.format(short_query=q)}]
    )
    rewritten_queries[q] = response.choices[0].message.content.strip()

for q in baseline_queries:
    rewritten = rewritten_queries[q]
    print("Original query:", q)
    print("Rewritten query:", rewritten)
    print("Baseline answer:", baseline_answers[q])
    print("Rewritten-query answer:", rag_chain.invoke(rewritten))
    print("=" * 60)
