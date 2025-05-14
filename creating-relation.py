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

def get_form_info(kg):
    cypher = """
      MATCH (anyChunk:Chunk) 
      WITH anyChunk LIMIT 1
      RETURN anyChunk { .names, .source, .formId, .cik, .cusip6 } as formInfo
    """
    form_info_list = kg.query(cypher)
    return form_info_list[0]['formInfo']

def create_form_node(kg, form_info):
    cypher = """
        MERGE (f:Form {formId: $formInfoParam.formId })
          ON CREATE 
            SET f.names = $formInfoParam.names
            SET f.source = $formInfoParam.source
            SET f.cik = $formInfoParam.cik
            SET f.cusip6 = $formInfoParam.cusip6
    """
    kg.query(cypher, params={'formInfoParam': form_info})

def count_form_nodes(kg):
    return kg.query("MATCH (f:Form) RETURN count(f) as formCount")

def link_chunks_by_section(kg, form_id, f10k_item):
    cypher = """
      MATCH (from_same_section:Chunk)
      WHERE from_same_section.formId = $formIdParam
        AND from_same_section.f10kItem = $f10kItemParam
      WITH from_same_section
        ORDER BY from_same_section.chunkSeqId ASC
      WITH collect(from_same_section) as section_chunk_list
        CALL apoc.nodes.link(
            section_chunk_list, 
            "NEXT", 
            {avoidDuplicates: true}
        )
      RETURN size(section_chunk_list)
    """
    return kg.query(cypher, params={'formIdParam': form_id, 'f10kItemParam': f10k_item})

def link_all_sections(kg, form_id, section_names):
    for section in section_names:
        link_chunks_by_section(kg, form_id, section)

def connect_chunks_to_form(kg):
    cypher = """
      MATCH (c:Chunk), (f:Form)
        WHERE c.formId = f.formId
      MERGE (c)-[newRelationship:PART_OF]->(f)
      RETURN count(newRelationship)
    """
    return kg.query(cypher)

def create_section_relationships(kg):
    cypher = """
      MATCH (first:Chunk), (f:Form)
      WHERE first.formId = f.formId
        AND first.chunkSeqId = 0
      WITH first, f
        MERGE (f)-[r:SECTION {f10kItem: first.f10kItem}]->(first)
      RETURN count(r)
    """
    return kg.query(cypher)

def get_first_chunk(kg, form_id, f10k_item):
    cypher = """
      MATCH (f:Form)-[r:SECTION]->(first:Chunk)
        WHERE f.formId = $formIdParam
            AND r.f10kItem = $f10kItemParam
      RETURN first.chunkId as chunkId, first.text as text
    """
    return kg.query(cypher, params={'formIdParam': form_id, 'f10kItemParam': f10k_item})[0]

def get_next_chunk(kg, chunk_id):
    cypher = """
      MATCH (first:Chunk)-[:NEXT]->(nextChunk:Chunk)
        WHERE first.chunkId = $chunkIdParam
      RETURN nextChunk.chunkId as chunkId, nextChunk.text as text
    """
    return kg.query(cypher, params={'chunkIdParam': chunk_id})[0]

def get_window_of_chunks(kg, chunk_id):
    cypher = """
        MATCH (c1:Chunk)-[:NEXT]->(c2:Chunk)-[:NEXT]->(c3:Chunk) 
            WHERE c2.chunkId = $chunkIdParam
        RETURN c1.chunkId, c2.chunkId, c3.chunkId
    """
    return kg.query(cypher, params={'chunkIdParam': chunk_id})

def get_variable_length_window(kg, chunk_id):
    cypher = """
      MATCH window=
          (:Chunk)-[:NEXT*0..1]->(c:Chunk)-[:NEXT*0..1]->(:Chunk) 
        WHERE c.chunkId = $chunkIdParam
      RETURN length(window)
    """
    return kg.query(cypher, params={'chunkIdParam': chunk_id})

def get_longest_window(kg, chunk_id):
    cypher = """
      MATCH window=
          (:Chunk)-[:NEXT*0..1]->(c:Chunk)-[:NEXT*0..1]->(:Chunk)
        WHERE c.chunkId = $chunkIdParam
      WITH window as longestChunkWindow 
          ORDER BY length(window) DESC LIMIT 1
      RETURN length(longestChunkWindow)
    """
    return kg.query(cypher, params={'chunkIdParam': chunk_id})

def main():
    # Load form info and create form node
    form_info = get_form_info(kg)
    create_form_node(kg, form_info)

    # Count form nodes
    form_count = count_form_nodes(kg)
    print(f"Number of Form nodes: {form_count[0]['formCount']}")

    # Link chunks by section
    section_names = ['item1', 'item1a', 'item7', 'item7a']
    for section in section_names:
        link_chunks_by_section(kg, form_info['formId'], section)

    # Connect chunks to form
    connect_chunks_to_form(kg)

if __name__ == "__main__":
  kg.refresh_schema()
  print(kg.schema)