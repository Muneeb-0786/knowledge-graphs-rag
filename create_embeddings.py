#!/usr/bin/env python
# coding: utf-8

# # Lesson 4: Constructing a Knowledge Graph from Text Documents

# <p style="background-color:#fd4a6180; padding:15px; margin-left:20px"> <b>Note:</b> This notebook takes about 30 seconds to be ready to use. Please wait until the "Kernel starting, please wait..." message clears from the top of the notebook before running any cells. You may start the video while you wait.</p>

# ### Import packages and set up Neo4j

# In[ ]:


from dotenv import load_dotenv
import os

# Common data processing
import json
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI


# Warning control
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def load_env_and_constants():
    load_dotenv('.env', override=True)
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_ENDPOINT = os.getenv('GEMINI_BASE_URL') + '/v1beta/models/embedding-001:embedContent'
    VECTOR_INDEX_NAME = 'form_10k_chunks'
    VECTOR_NODE_LABEL = 'Chunk'
    VECTOR_SOURCE_PROPERTY = 'text'
    VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'
    return {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_ENDPOINT": GEMINI_ENDPOINT,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "VECTOR_NODE_LABEL": VECTOR_NODE_LABEL,
        "VECTOR_SOURCE_PROPERTY": VECTOR_SOURCE_PROPERTY,
        "VECTOR_EMBEDDING_PROPERTY": VECTOR_EMBEDDING_PROPERTY,
    }

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )

def load_form10k_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def split_form10k_data_from_file(file, text_splitter):
    chunks_with_metadata = []
    file_as_object = load_form10k_json(file)
    for item in ['item1','item1a','item7','item7a']:
        print(f'Processing {item} from {file}') 
        item_text = file_as_object[item]
        item_text_chunks = text_splitter.split_text(item_text)
        chunk_seq_id = 0
        for chunk in item_text_chunks[:20]:
            form_id = file[file.rindex('/') + 1:file.rindex('.')]
            chunks_with_metadata.append({
                'text': chunk, 
                'f10kItem': item,
                'chunkSeqId': chunk_seq_id,
                'formId': f'{form_id}',
                'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                'names': file_as_object['names'],
                'cik': file_as_object['cik'],
                'cusip6': file_as_object['cusip6'],
                'source': file_as_object['source'],
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata

def create_kg_connection(cfg):
    return Neo4jGraph(
        url=cfg["NEO4J_URI"],
        username=cfg["NEO4J_USERNAME"],
        password=cfg["NEO4J_PASSWORD"],
        database=cfg["NEO4J_DATABASE"]
    )

def create_chunk_nodes(kg, chunks, merge_chunk_node_query):
    node_count = 0
    for chunk in chunks:
        print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
        kg.query(merge_chunk_node_query, params={'chunkParam': chunk})
        node_count += 1
    print(f"Created {node_count} nodes")

def create_uniqueness_constraint(kg):
    kg.query("""
    CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
        FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
    """)

def create_vector_index(kg):
    kg.query("""
        CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
        FOR (c:Chunk) ON (c.textEmbedding) 
        OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
        }}
    """)

def populate_embeddings(kg, gemini_api_key, gemini_endpoint):
    kg.query("""
        MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
        WITH chunk, genai.vector.encode(
            chunk.text, 
            "Gemini", 
            {
                token: $geminiApiKey, 
                endpoint: $geminiEndpoint
            }) AS vector
        CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
        """, 
        params={"geminiApiKey": gemini_api_key, "geminiEndpoint": gemini_endpoint} )

def neo4j_vector_search(kg, question, gemini_api_key, gemini_endpoint, vector_index_name, top_k=10):
    vector_search_query = """
        WITH genai.vector.encode(
            $question, 
            "Gemini", 
            {
                token: $geminiApiKey,
                endpoint: $geminiEndpoint
            }) AS question_embedding
        CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) yield node, score
        RETURN score, node.text AS text
    """
    similar = kg.query(vector_search_query, 
        params={
            'question': question, 
            'geminiApiKey': gemini_api_key,
            'geminiEndpoint': gemini_endpoint,
            'index_name': vector_index_name, 
            'top_k': top_k})
    return similar

def get_neo4j_vector_store(cfg):
    return Neo4jVector.from_existing_graph(
        embedding=GoogleGenerativeAIEmbeddings(google_api_key=cfg["GEMINI_API_KEY"]),
        url=cfg["NEO4J_URI"],
        username=cfg["NEO4J_USERNAME"],
        password=cfg["NEO4J_PASSWORD"],
        index_name=cfg["VECTOR_INDEX_NAME"],
        node_label=cfg["VECTOR_NODE_LABEL"],
        text_node_properties=[cfg["VECTOR_SOURCE_PROPERTY"]],
        embedding_node_property=cfg["VECTOR_EMBEDDING_PROPERTY"],
    )

def get_chain(retriever):
    return RetrievalQAWithSourcesChain.from_chain_type(
        ChatGoogleGenerativeAI(google_api_key=os.getenv('GEMINI_API_KEY'), temperature=0), 
        chain_type="stuff", 
        retriever=retriever
    )

def prettychain(chain, question: str) -> str:
    response = chain({"question": question}, return_only_outputs=True)
    print(textwrap.fill(response['answer'], 60))

def main():
    cfg = load_env_and_constants()
    text_splitter = get_text_splitter()
    first_file_name = "./data/form10k/0000950170-23-027948.json"
    first_file_chunks = split_form10k_data_from_file(first_file_name, text_splitter)

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

    kg = create_kg_connection(cfg)
    create_uniqueness_constraint(kg)
    create_chunk_nodes(kg, first_file_chunks, merge_chunk_node_query)
    create_vector_index(kg)
    populate_embeddings(kg, cfg["GEMINI_API_KEY"], cfg["GEMINI_ENDPOINT"])
    kg.refresh_schema()
    print(kg.schema)

    # Example vector search
    search_results = neo4j_vector_search(
        kg,
        'In a single sentence, tell me about Netapp.',
        cfg["GEMINI_API_KEY"],
        cfg["GEMINI_ENDPOINT"],
        cfg["VECTOR_INDEX_NAME"]
    )
    print(search_results[0])

    # LangChain RAG workflow
    neo4j_vector_store = get_neo4j_vector_store(cfg)
    retriever = neo4j_vector_store.as_retriever()
    chain = get_chain(retriever)

    # Example questions
    prettychain(chain, "What is Netapp's primary business?")
    prettychain(chain, "Where is Netapp headquartered?")
    prettychain(chain, "Tell me about Netapp. Limit your answer to a single sentence.")
    prettychain(chain, "Tell me about Apple. Limit your answer to a single sentence.")
    prettychain(chain, "Tell me about Apple. Limit your answer to a single sentence. If you are unsure about the answer, say you don't know.")

    # Add your own question here
    prettychain(chain, "ADD YOUR OWN QUESTION HERE")

if __name__ == "__main__":
    main()

