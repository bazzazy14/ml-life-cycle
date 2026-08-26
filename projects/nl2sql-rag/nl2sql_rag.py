# Cleaned from the completed Break Through Tech NL2SQL notebook.

from operator import itemgetter
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

DB_URI = "sqlite:///data/Chinook.db"
db = SQLDatabase.from_uri(DB_URI)
llm = ChatOpenAI(model="gpt-4o")

# Step 1: generate SQL from a natural-language question.
write_query = create_sql_query_chain(llm, db)

# Step 2: normalize the model response into executable SQL only.
def clean_sql_query(response: str) -> str:
    prompt = f"""
Here is a response from an LLM that contains a SQL query:

{response}

Extract only the SQL query from the response.
Do not include markdown code blocks, explanations, introductions, comments,
or any additional text. Return only directly executable SQL.
"""
    clean_response = llm.invoke([HumanMessage(content=prompt)])
    return clean_response.content

# Step 3: execute the generated query against the relational database.
execute_query = QuerySQLDatabaseTool(db=db)
query_chain = write_query | clean_sql_query | execute_query

# Step 4: synthesize a natural-language response from the SQL result.
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result,
answer the user's question clearly and concisely.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:
"""
)

complete_chain = (
    RunnablePassthrough.assign(query=write_query | clean_sql_query)
    .assign(result=itemgetter("query") | execute_query)
    | answer_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    questions = [
        "How many employees are there?",
        "Who are the top 5 customers on the basis of spending?",
        "What are the top 3 albums by sales?",
    ]

    for question in questions:
        print(f"Question: {question}")
        print(complete_chain.invoke({"question": question}))
        print("-" * 60)
