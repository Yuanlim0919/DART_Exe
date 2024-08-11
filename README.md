# DART_Exe
This is the exercise of DART Orientation, it contain two parts: RAG by vector and RAG by Graph

## Requirements
Set up the system on local, and try to insert a document by your own and do RAG

## System components
- Ollama: For hosting LLM
- LlamaIndex: For insert data into Neo4j
- LangChain: For implementation of RAG pipeline
- Neo4j: Graph Database
- Qdrant: Vector Database

## steps
### Hosting LLM on local environment
1. Download Ollama from its official website
2. In command prompt, enter `ollama` to verify install was succeeded
3. Download your desired model by command `ollama run <model name>`. In this implementation, I've used `llama3.1`.

### RAG by Vector database

### RAG by Graph database
1. Execute docker compose file
`docker compose up -d`
2. Afterwards, visit `localhost:7474` to ensure Neo4j was launched successfully
3. Create a directory and insert document you want to use into it (or replace the document in `example_doc`)
4. Insert the data into Neo4j
`python ./GraphRAG/GraphRAG.py -d <directory path stores your document>`
5. After data insertion completed, ask your question by
`python ./GraphRAG/GraphRAG.py -q True -i <directory name stores your document>`

## To Do
1. Fuzzy search of Neo4j
2. Output quality enhancement
    - by base model
    - by prompting
3. GPU supporting
