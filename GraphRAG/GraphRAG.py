import os
import json
import argparse
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama as ChatOllama
from langchain.prompts import PromptTemplate
from llama_index.legacy.llms.ollama import Ollama as OllamaIndex
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext
from llama_index.core.prompts.base import PromptTemplate as LlamaIndexPromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.legacy.graph_stores import Neo4jGraphStore

load_dotenv()

class GraphRAG:
    def __init__(self):
        self.llm = OllamaIndex(
            model='llama3.1',
            temperature=0.2,
            request_timeout=300
        )
        self.embedding_model = HuggingFaceEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        self.graph_store = Neo4jGraphStore(
            username=os.environ['NEO4J_USERNAME'],
            password=os.environ['NEO4J_PASSWORD'],
            url=os.environ['NEO4J_URI'],
            database=os.environ['NEO4J_DATABASE']
        )
        kg_entity_extract_prompt = (
            "Some text is provided below. Given the text, extract up to "
            "{max_knowledge_triplets} "
            "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
            "If the text was provided with transcript format, please extract the triplets based on the person mentioned\n"
            "---------------------\n"
            "Example:"
            "Text: 'Speaker1: I have a pen'\n"
            "Triplets:\n(Speaker1, have, pen)\n"
            "Text: 'Speaker2: I introduce an event'\n"
            "Triplets:\n(Speaker2, introduce, event)\n"
            "Text: Alice is Bob's mother.\n"
            "Triplets:\n(Alice, is mother of, Bob)\n"
            "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
            "Triplets:\n"
            "(Philz, is, coffee shop)\n"
            "(Philz, founded in, Berkeley)\n"
            "(Philz, founded in, 1982)\n"
            "---------------------\n"
            "Text: {text}\n"
            "Triplets:\n"
            "Triplets:\n"
        )
        self.kg_prompt_template = LlamaIndexPromptTemplate(
            template=kg_entity_extract_prompt,
            prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
        )
        pass

    def insert_doc(self,doc_path):
        self.graph_store.node_label = doc_path.split('/')[-1]
        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store,
        )
        document = SimpleDirectoryReader(doc_path).load_data(show_progress=True)
        kg_index = KnowledgeGraphIndex.from_documents(
            documents=document,
            storage_context=self.storage_context,
            kg_triplet_extract_template=self.kg_prompt_template,
            max_triplets_per_chunk=20,
            llm=self.llm,
            embed_model=self.embedding_model,
        )
        pass

    def langchain_call(self,task,query,knowledge=None):
        properties_extract_prompt = PromptTemplate(
            template='''
            From the input sentence, please extract the nouns existed in the sentence, and return it in a list.
            The noun may be contain multiple words, but it should be considered as a single noun.
            For example, 'New York', 'LeBron James' should be considered as a single noun.
            If the sentence contains multiple sentences, please extract the nouns from all sentences.
            In the reply you only need to return the list, and without additional information.
            For the list, you should follow the format below and do not change the input context:
            '["noun1", "noun2", "noun3", ...]'
            sentence:{question}
            ''',
            input_variables=['question']
        )
        llm = ChatOllama(
            model='llama3.1',
            temperature=0.,
        )
        qa_prompt = PromptTemplate(
            template='''
            Please answer the question only according to provided knowledges and without prior knowledges.
            If the question asked is exceeded the provided knowledges, do not answer the question.
            Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
            Please try to answer the question more extensively by giving examples provided in knowledge or giving elabotation.
            When the extracted knowledge is too much, try to figure out the most important part and answer the question.
            
            Question: {question}
            Knowledge: {knowledge}
            ''',
            input_variables=['question','knowledge']
        )
        if task == 'properties':
            properties_extract_chain = properties_extract_prompt | llm
            properties = properties_extract_chain.invoke({'question': query})
            breakpoint()
            properties = json.loads(properties)
            return properties
        elif task == 'qa':
            qa_chain = qa_prompt | llm
            answer = qa_chain.invoke({'question': query,'knowledge':knowledge})
            #answer = json.loads(answer)
            return answer

    def collect_subjects(self,d,subject_list=None):
        if subject_list is None:
            subject_list = []
        
        # Add the value of the current 'subject' key to the list
        if 'subject' in d and d['subject']:
            subject_list.append(d['subject'])
        
        # Recursively collect 'subject' values from the nested 'object'
        if 'object' in d and isinstance(d['object'], dict):
            self.collect_subjects(d['object'], subject_list)
        
        subject_list = [item for sublist in subject_list for item in (sublist if isinstance(sublist, list) else [sublist])]

        return subject_list

    def query(self,query,doc_id):
        self.graph_store.node_label = doc_id
        properties = self.langchain_call(task='properties',query=query,)
        #subjects = self.collect_subjects(properties)
        knowledge = self.graph_store.get_rel_map(properties)
        answer = self.langchain_call(query=query,knowledge=knowledge,task='qa')
        return answer


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d','--doc_path', type=str, help='Path to the document to be inserted')
    group.add_argument('-q','--query', type=str, help='Query to be run')
    parser.add_argument('-i','--id', type=str, help='Document ID')
    args = parser.parse_args()

    rag = GraphRAG()
    if args.doc_path:
        rag.insert_doc(args.doc_path)
    if args.query:
        while True:
            question = input("Enter question: ")
            if question == 'exit':
                break
            ans = rag.query(query=question,doc_id=args.id)
            print(ans)
main()