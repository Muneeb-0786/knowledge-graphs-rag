# Import libraries and organize them by category
# Environment & configuration
from dotenv import load_dotenv
import os
import warnings

# Data processing
import json
import textwrap
import requests

# Neo4j
from neo4j import GraphDatabase

# LangChain components
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Google Generative AI
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Check for required environment variables
if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_KEY]):
    raise EnvironmentError("Missing one or more required environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_KEY")

# Global constants for Neo4j vector search
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'
GEMINI_ENDPOINT='https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent'

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)


first_file_name = "./data/form10k/0000950170-23-027948.json"
first_file_as_object = json.load(open(first_file_name))
type(first_file_as_object)
for k,v in first_file_as_object.items():
    print(k, type(v))

item1_text = first_file_as_object['item1']
item1_text[0:1500]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)
item1_text_chunks = text_splitter.split_text(item1_text)


def split_form10k_data_from_file(file):
    chunks_with_metadata = []  # use this to accumulate chunk records
    file_as_object = json.load(open(file))  # open the json file
    for item in ['item1', 'item1a', 'item7', 'item7a']:  # pull these keys from the json
        print(f'Processing {item} from {file}')
        item_text = file_as_object[item]  # grab the text of the item
        item_text_chunks = text_splitter.split_text(item_text)  # split the text into chunks
        chunk_seq_id = 0
        for chunk in item_text_chunks[:20]:  # only take the first 20 chunks
            form_id = file[file.rindex('/') + 1:file.rindex('.')]  # extract form id from file name
            # finally, construct a record with metadata and the chunk text
            chunks_with_metadata.append({
                'text': chunk,
                # metadata from looping...
                'f10kItem': item,
                'chunkSeqId': chunk_seq_id,
                # constructed metadata...
                'formId': f'{form_id}',  # pulled from the filename
                'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                # metadata from file...
                'names': file_as_object['names'],
                'cik': file_as_object['cik'],
                'cusip6': file_as_object['cusip6'],
                'source': file_as_object['source'],
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata

first_file_chunks = split_form10k_data_from_file(first_file_name)

first_file_chunks[0]

merge_chunk_node_query = """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    ON CREATE SET 
        mergedChunk.names = $chunkParam.names,
        mergedChunk.formId = $chunkParam.formId, 
        mergedChunk.cik = $chunkParam.cik, 
        mergedChunk.cusip6 = $chunkParam.cusip6, 
        mergedChunk.source = $chunkParam.source, 
        mergedChunk.f10kItem = $chunkParam.f10kItem, 
        mergedChunk.chunkSeqId = $chunkParam.chunkSeqId, 
        mergedChunk.text = $chunkParam.text
RETURN mergedChunk
"""

kg.query("""
CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
""")
print("Created unique constraint for chunkId", kg.query("SHOW INDEXES"))

node_count = 0
for chunk in first_file_chunks:
    print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
    kg.query(merge_chunk_node_query, 
            params={
                'chunkParam': chunk
            })
    node_count += 1
print(f"Created {node_count} nodes")

print(kg.query("""
         MATCH (n)
         RETURN count(n) as nodeCount
         """))

kg.query("""
         CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'    
         }}
""")

print("Created vector index", kg.query("SHOW INDEXES"))

# Add Gemini embedding function in Python
def get_gemini_embedding(text, api_key):
    url = GEMINI_ENDPOINT + f"?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    embedding = response.json()["embedding"]["values"]
    return embedding

# Remove Cypher embedding code, replace with Python embedding + update
for chunk in first_file_chunks:
    embedding = get_gemini_embedding(chunk['text'], GEMINI_API_KEY)
    kg.query(
        """
        MATCH (c:Chunk {chunkId: $chunkId})
        SET c.textEmbedding = $embedding
        """,
        params={"chunkId": chunk['chunkId'], "embedding": embedding}
    )

kg.refresh_schema()
print(kg.schema)

def neo4j_vector_search(question):
    """Search for similar nodes using the Neo4j vector index (Gemini embeddings)"""
    # Generate embedding for the question in Python
    question_embedding = get_gemini_embedding(question, GEMINI_API_KEY)
    vector_search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $question_embedding) yield node, score
        RETURN score, node.text AS text
    """
    similar = kg.query(vector_search_query,
                       params={
                           'question_embedding': question_embedding,
                           'index_name': VECTOR_INDEX_NAME,
                           'top_k': 10
                       })
    return similar

search_results = neo4j_vector_search(
    'In a single sentence, tell me about Netapp.'
)

print("Search results:", search_results[0])

def setup_langchain_qa():
    """Set up a LangChain QA system with Neo4j Vector Store"""
    # Initialize the vector store from existing Neo4j graph
    neo4j_vector_store = Neo4jVector.from_existing_graph(
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",
            google_api_key=GEMINI_API_KEY),
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=VECTOR_INDEX_NAME,
        node_label=VECTOR_NODE_LABEL,
        text_node_properties=[VECTOR_SOURCE_PROPERTY],
        embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
    )
    
    # Create a retriever from the vector store
    retriever = neo4j_vector_store.as_retriever()
    
    # Create a QA chain with the proper LangChain model wrapper
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
    
    # Create a QA chain
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff", 
        retriever=retriever
    )
qa_chain = setup_langchain_qa()
question = "What is the business of Netapp?"
response = qa_chain({"question": question})
print("Response:", response['answer'])