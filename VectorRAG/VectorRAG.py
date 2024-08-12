import os
import json
import argparse
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama as ChatOllama
from langchain.prompts import PromptTemplate
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy import SimpleDirectoryReader, StorageContext
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.llms.ollama import Ollama as OllamaIndex
from llama_index.legacy.vector_stores import QdrantVectorStore
from pydantic import BaseModel
from qdrant_client import QdrantClient

load_dotenv()

class Response(BaseModel):
    search_result: str 
    source: str

class VectorRAG:
    def __init__(self,collection_name):
        self.llm = OllamaIndex(
            model='llama3.1',
            temperature=0.2,
            request_timeout=300
        )
        self.embedding_model = HuggingFaceEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        qdrant_client = QdrantClient(
            url=os.environ['QDRANT_URI'],\
            timeout=300
        )
        self.vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name
        )
        self.collection_name = collection_name
        self.chunk_size = 100
        self.top_k = 3
        pass

    def insert_doc(self,doc_path):
        from llama_index.legacy.indices import VectorStoreIndex
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
        )
        document = SimpleDirectoryReader(doc_path).load_data(show_progress=True)
        service_context = ServiceContext.from_defaults(
            llm = self.llm,
            embed_model = self.embedding_model,
            chunk_size = self.chunk_size,
            chunk_overlap=100
        )
        vector_index = VectorStoreIndex.from_documents(
            documents=document,
            storage_context=self.storage_context,
            service_context=service_context
        )
        pass

    def langchain_call(self,query,knowledge=None):
        llm = ChatOllama(
            model='llama3.1',
            temperature=0.,
        )
        qa_prompt = PromptTemplate(
            template='''
            Please answer the question only according to provided knowledges.
            Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
            Please try to answer the question more extensively by giving examples provided in knowledge or giving elabotation.
            When the extracted knowledge is too much, try to figure out the most important part and answer the question.
            
            Question: {question}
            Knowledge: {knowledge}
            ''',
            input_variables=['question','knowledge']
        )
        
        qa_chain = qa_prompt | llm
        answer = qa_chain.invoke({'question': query,'knowledge':knowledge})
            #answer = json.loads(answer)
        return answer

    def query_index(self):
        from llama_index.legacy.indices import VectorStoreIndex
        client = QdrantClient(
            url=os.environ['QDRANT_URI'],
            timeout=300
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name
        )
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=100
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context
        )
        return index

    def query(self,query):
        query_engine = self.query_index().as_query_engine(
            similarity_top_k = self.top_k,
            response_mode='tree_summarize'
        )
        response = query_engine.query(query)
        response_object = Response(
            search_result=str(response).strip(), 
            source=[response.metadata[k]["file_path"] for k in response.metadata.keys()][0]
        )
        print(response_object)
        return response_object.search_result

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d','--doc_path', type=str, help='Path to the document to be inserted')
    group.add_argument('-q','--query', type=str, help='Query to be run')
    parser.add_argument('-i','--id', type=str, help='Document ID')
    args = parser.parse_args()

    if args.doc_path:
        collection_name = args.doc_path.split('/')[-1]
        rag = VectorRAG(collection_name)
        rag.insert_doc(args.doc_path)
    if args.query:
        collection_name = args.id
        rag = VectorRAG(collection_name)
        while True:
            question = input("Enter question: ")
            if question == 'exit':
                break
            knowledge = rag.query(query=question)
            ans = rag.langchain_call(query=question,knowledge=knowledge)
            print(ans)
main()