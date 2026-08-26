# Break Through Tech AI/ML Portfolio

A collection of applied machine learning and generative AI projects completed through the Break Through Tech AI Program and Cornell University Machine Learning Foundations curriculum.

## Featured Project

### Airbnb Price Classification
Built a binary classification pipeline over 28,022 NYC Airbnb listings. After leakage-aware feature selection and preprocessing, I tuned Logistic Regression with 5-fold cross-validation and trained a 5,185-parameter neural network. The neural network improved test accuracy from 82.96% to 85.11% and F1 from 0.612 to 0.687.

[View the Airbnb project](projects/airbnb-price-classification/)

## Projects

| Project | Focus | Tools |
| --- | --- | --- |
| [Airbnb Price Classification](projects/airbnb-price-classification/) | Classification, feature engineering, cross-validation, neural networks | Python, scikit-learn, TensorFlow/Keras, Pandas |
| [Modeling Dataset Preparation](projects/modeling-dataset-preparation/) | Data cleaning, missing values, outliers, feature preparation | Python, Pandas, NumPy |
| [Decision Tree Classification](projects/decision-tree-classification/) | Decision trees, one-hot encoding, model evaluation | Python, scikit-learn, Pandas |
| [Document RAG](projects/document-rag/) | Chunking, embeddings, vector retrieval, grounded generation | LangChain, Chroma, OpenAI API |
| [Natural Language to SQL RAG](projects/nl2sql-rag/) | NL2SQL, database retrieval, answer synthesis | LangChain, SQLAlchemy, SQLite, OpenAI API |
| [MCP Server](projects/mcp-server/) | Model Context Protocol tools and agent integration | Python, FastMCP, OpenAI Agents SDK |
| [Agentic Pitch Clinic](projects/agentic-pitch-clinic/) | Routing, evaluator-optimizer loops, structured outputs, MCP | OpenAI Agents SDK, Pydantic, FastMCP |
| [Multimodal Document Extraction](projects/multimodal-document-extraction/) | Vision-language extraction and structured data | OpenAI API, Python, Pandas |

## Separate Portfolio Repository

A cleaned, recruiter-facing version of the multimodal project is also published separately at [multimodal-document-intelligence](https://github.com/bazzazy14/multimodal-document-intelligence).

## Notes

These notebooks originated from Break Through Tech coursework and labs. The code, analysis, model choices, experiments, and completed implementations in the notebooks reflect my work. Some notebooks depend on course-provided datasets or local assets that are not included here.
