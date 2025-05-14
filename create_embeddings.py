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

GEMINI_ENDPOINT='https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent'

def load_env_and_constants():
    load_dotenv('.env', override=True)
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    VECTOR_INDEX_NAME = 'form_10k_chunks'
    VECTOR_NODE_LABEL = 'Chunk'
    VECTOR_SOURCE_PROPERTY = 'text'
    VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'
    # Add validation for required Neo4j credentials
    missing = []
    if not NEO4J_URI:
        missing.append("NEO4J_URI")
    if not NEO4J_USERNAME:
        missing.append("NEO4J_USERNAME")
    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")
    if missing:
        raise ValueError(
            f"Missing required Neo4j environment variables: {', '.join(missing)}. "
            "Please check your .env file."
        )
    return {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
        "GEMINI_API_KEY": GEMINI_API_KEY,
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
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'    
        }}
    """)

def populate_embeddings(kg, gemini_api_key, gemini_endpoint):
    # 1. Get all Chunk nodes without embeddings
    chunks = kg.query("""
        MATCH (chunk:Chunk)
        WHERE chunk.textEmbedding IS NULL
        RETURN chunk.chunkId AS chunkId, chunk.text AS text
    """)
    if not chunks:
        print("No chunks found that need embeddings.")
        return

    # 2. Set up the embedding model
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )

    # 3. For each chunk, generate embedding and update node
    for chunk in chunks:
        chunk_id = chunk["chunkId"]
        text = chunk["text"]
        try:
            embedding = embedder.embed_query(text)
            # Update the node with the embedding
            kg.query(
                """
                MATCH (c:Chunk {chunkId: $chunkId})
                SET c.textEmbedding = $embedding
                """,
                params={"chunkId": chunk_id, "embedding": embedding}
            )
            print(f"Set embedding for chunk {chunk_id}")
        except Exception as e:
            print(f"Failed to embed chunk {chunk_id}: {e}")

def chunks_exist_for_form(kg, form_id):
    result = kg.query(
        "MATCH (c:Chunk {formId: $formId}) RETURN count(c) AS count",
        params={"formId": form_id}
    )
    return result[0]['count'] > 0

def main():
    cfg = load_env_and_constants()
    text_splitter = get_text_splitter()
    first_file_name = "./data/form10k/0000950170-23-027948.json"
    # Extract form_id from filename (same logic as in split_form10k_data_from_file)
    form_id = first_file_name[first_file_name.rindex('/') + 1:first_file_name.rindex('.')]
    kg = create_kg_connection(cfg)
    create_uniqueness_constraint(kg)
    if chunks_exist_for_form(kg, form_id):
        print(f"Chunks for formId '{form_id}' already exist. Skipping chunk creation and embedding population.")
    else:
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
        create_chunk_nodes(kg, first_file_chunks, merge_chunk_node_query)
        create_vector_index(kg)
        populate_embeddings(kg, cfg["GEMINI_API_KEY"], GEMINI_ENDPOINT)
    kg.refresh_schema()
    print(kg.schema)

if __name__ == "__main__":
    main()

