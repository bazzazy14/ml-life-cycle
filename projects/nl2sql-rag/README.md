# Natural Language to SQL RAG

An LLM-powered structured-data retrieval pipeline that translates natural-language questions into SQL, executes the query against a relational database, and converts raw results into user-friendly answers.

## What it demonstrates
- LangChain SQLDatabase integration
- SQL generation from natural language
- SQL cleaning/validation before execution
- Query execution with QuerySQLDatabaseTool
- SQL joins and aggregations
- Answer synthesis from database results
- End-to-end runnable chain composition

[Review the source](nl2sql_rag.py)

> The Chinook SQLite database used in the original exercise is a course-provided/local asset and is not redistributed here.
