# Import libraries and organize them by category
# Environment & configuration
from dotenv import load_dotenv
import os
import warnings

# Data processing
import json
import textwrap
import requests
import re

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

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

import csv

all_form13s = []

def read_form13_csv(filepath='./data/form13.csv'):
    all_form13s = []
    with open(filepath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            all_form13s.append(row)
    return all_form13s

def create_company_node(kg, form13):
    cypher = """
    MERGE (com:Company {cusip6: $cusip6})
      ON CREATE
        SET com.companyName = $companyName,
            com.cusip = $cusip
    """
    kg.query(cypher, params={
        'cusip6': form13['cusip6'],
        'companyName': form13['companyName'],
        'cusip': form13['cusip']
    })

def update_company_names(kg):
    cypher = """
      MATCH (com:Company), (form:Form)
        WHERE com.cusip6 = form.cusip6
      SET com.names = form.names
    """
    kg.query(cypher)

def create_filed_relationship(kg):
    cypher = """
      MATCH (com:Company), (form:Form)
        WHERE com.cusip6 = form.cusip6
      MERGE (com)-[:FILED]->(form)
    """
    kg.query(cypher)

def create_manager_node(kg, form13):
    cypher = """
      MERGE (mgr:Manager {managerCik: $managerParam.managerCik})
        ON CREATE
            SET mgr.managerName = $managerParam.managerName,
                mgr.managerAddress = $managerParam.managerAddress
    """
    kg.query(cypher, params={'managerParam': form13})

def create_manager_constraint(kg):
    cypher = """
    CREATE CONSTRAINT unique_manager 
      IF NOT EXISTS
      FOR (n:Manager) 
      REQUIRE n.managerCik IS UNIQUE
    """
    kg.query(cypher)

def create_manager_fulltext_index(kg):
    cypher = """
    CREATE FULLTEXT INDEX fullTextManagerNames
      IF NOT EXISTS
      FOR (mgr:Manager) 
      ON EACH [mgr.managerName]
    """
    kg.query(cypher)

def create_all_manager_nodes(kg, all_form13s):
    cypher = """
      MERGE (mgr:Manager {managerCik: $managerParam.managerCik})
        ON CREATE
            SET mgr.managerName = $managerParam.managerName,
                mgr.managerAddress = $managerParam.managerAddress
    """
    for form13 in all_form13s:
        kg.query(cypher, params={'managerParam': form13})

def create_owns_stock_relationships(kg, all_form13s):
    cypher = """
    MATCH (mgr:Manager {managerCik: $ownsParam.managerCik}), 
            (com:Company {cusip6: $ownsParam.cusip6})
    MERGE (mgr)-[owns:OWNS_STOCK_IN { 
        reportCalendarOrQuarter: $ownsParam.reportCalendarOrQuarter 
        }]->(com)
      ON CREATE
        SET owns.value  = toFloat($ownsParam.value), 
            owns.shares = toInteger($ownsParam.shares)
    """
    for form13 in all_form13s:
        kg.query(cypher, params={'ownsParam': form13})

def get_first_chunk_id(kg):
    cypher = """
        MATCH (chunk:Chunk)
        RETURN chunk.chunkId as chunkId LIMIT 1
    """
    chunk_rows = kg.query(cypher)
    return chunk_rows[0]['chunkId'] if chunk_rows else None

def get_company_name_from_chunk(kg, chunk_id):
    cypher = """
    MATCH (:Chunk {chunkId: $chunkIdParam})-[:PART_OF]->(f:Form),
        (com:Company)-[:FILED]->(f)
    RETURN com.companyName as name
    """
    return kg.query(cypher, params={'chunkIdParam': chunk_id})

def get_number_of_investors(kg, chunk_id):
    cypher = """
    MATCH (:Chunk {chunkId: $chunkIdParam})-[:PART_OF]->(f:Form),
            (com:Company)-[:FILED]->(f),
            (mgr:Manager)-[:OWNS_STOCK_IN]->(com)
    RETURN com.companyName, 
            count(mgr.managerName) as numberOfinvestors 
    LIMIT 1
    """
    return kg.query(cypher, params={'chunkIdParam': chunk_id})

def get_investment_sentences(kg, chunk_id):
    cypher = """
        MATCH (:Chunk {chunkId: $chunkIdParam})-[:PART_OF]->(f:Form),
            (com:Company)-[:FILED]->(f),
            (mgr:Manager)-[owns:OWNS_STOCK_IN]->(com)
        RETURN mgr.managerName + " owns " + owns.shares + 
            " shares of " + com.companyName + 
            " at a value of $" + 
            apoc.number.format(toInteger(owns.value)) AS text
        LIMIT 10
    """
    return kg.query(cypher, params={'chunkIdParam': chunk_id})

def extract_address_nodes(form13):
    address_nodes = []
    
    for form in form13:
        address = form.get('managerAddress')
        company_address = form.get('companyAddress')
        if address and company_address not in address_nodes:
            address_nodes.append({
                'address': address,
              
            })
    return address_nodes

def extract_addresses_from_form13s(form13s):
    """
    Extracts unique addresses from manager and company fields.
    Returns a list of dicts: { 'address': ..., 'type': 'Manager'/'Company', 'ref_id': ... }
    """
    addresses = []
    seen = set()
    for form in form13s:
        # Manager address
        mgr_addr = form.get('managerAddress')
        mgr_cik = form.get('managerCik')
        if mgr_addr and (mgr_addr, 'Manager', mgr_cik) not in seen:
            addresses.append({'address': mgr_addr, 'type': 'Manager', 'ref_id': mgr_cik})
            seen.add((mgr_addr, 'Manager', mgr_cik))
        # Company address
        comp_addr = form.get('companyAddress')
        cusip6 = form.get('cusip6')
        if comp_addr and (comp_addr, 'Company', cusip6) not in seen:
            addresses.append({'address': comp_addr, 'type': 'Company', 'ref_id': cusip6})
            seen.add((comp_addr, 'Company', cusip6))
    return addresses

def geocode_address(address):
    """
    Geocode address using OpenStreetMap Nominatim API.
    Returns dict with city, state, country, latitude, longitude if found.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "addressdetails": 1,
        "limit": 1
    }
    headers = {
        "User-Agent": "rag-graph-geocoder/1.0"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            result = data[0]
            address_details = result.get('address', {})
            city = address_details.get('city') or address_details.get('town') or address_details.get('village') or ''
            state = address_details.get('state', '')
            country = address_details.get('country', '')
            lat = float(result['lat'])
            lon = float(result['lon'])
            return {
                'city': city,
                'state': state,
                'country': country,
                'latitude': lat,
                'longitude': lon
            }
    return None

def create_address_node_and_relationship(kg, address_info):
    """
    Create (:Address) node and connect to Manager/Company with [:LOCATED_AT].
    """
    # Create Address node
    cypher_address = """
    MERGE (addr:Address {fullAddress: $address})
      ON CREATE SET addr.city = $city, addr.state = $state, addr.country = $country,
                    addr.latitude = $latitude, addr.longitude = $longitude
    """
    kg.query(cypher_address, params={
        'address': address_info['address'],
        'city': address_info.get('city', ''),
        'state': address_info.get('state', ''),
        'country': address_info.get('country', ''),
        'latitude': address_info.get('latitude', None),
        'longitude': address_info.get('longitude', None)
    })
    # Create LOCATED_AT relationship
    if address_info['type'] == 'Manager':
        cypher_rel = """
        MATCH (mgr:Manager {managerCik: $ref_id}), (addr:Address {fullAddress: $address})
        MERGE (mgr)-[:LOCATED_AT]->(addr)
        """
    else:
        cypher_rel = """
        MATCH (com:Company {cusip6: $ref_id}), (addr:Address {fullAddress: $address})
        MERGE (com)-[:LOCATED_AT]->(addr)
        """
    kg.query(cypher_rel, params={
        'ref_id': address_info['ref_id'],
        'address': address_info['address']
    })

def simplify_address(address):
    """
    Simplify address by removing suite/floor/unit, PO boxes, numbered floors, and extra commas.
    """
    # Remove PO Box
    address = re.sub(r'(?i)\bP\.?O\.?\s*Box\s*\d+\b', '', address)
    address = re.sub(r'(?i)\bPOST OFFICE BOX\s*\d+\b', '', address)
    # Remove suite/floor/unit (case-insensitive)
    address = re.sub(r',?\s*(suite|ste\.?|floor|fl\.?|unit|apt\.?|apartment|#)\s*\w+', '', address, flags=re.IGNORECASE)
    # Remove ordinal floors (e.g., "Ninth Floor", "9th Floor")
    address = re.sub(r',?\s*\b(\d+(st|nd|rd|th)|[A-Za-z]+)\s+Floor\b', '', address, flags=re.IGNORECASE)
    # Remove extra commas and whitespace
    address = re.sub(r'\s+', ' ', address)
    address = re.sub(r',\s*,', ',', address)
    address = re.sub(r',\s*$', '', address)
    address = address.strip()
    # Remove leading/trailing commas
    address = re.sub(r'^,+', '', address)
    address = re.sub(r',+$', '', address)
    return address

def expand_with_address_nodes(kg, all_form13s):
    """
    Extract, geocode, and expand the graph with Address nodes and LOCATED_AT relationships.
    If geocoding fails, retry with a simplified address.
    """
    addresses = extract_addresses_from_form13s(all_form13s)
    for addr in addresses:
        geo = geocode_address(addr['address'])
        if geo:
            addr.update(geo)
            create_address_node_and_relationship(kg, addr)
        else:
            # Try with simplified address
            simple_addr = simplify_address(addr['address'])
            if simple_addr != addr['address']:
                geo_simple = geocode_address(simple_addr)
                if geo_simple:
                    addr.update(geo_simple)
                    create_address_node_and_relationship(kg, addr)
                    print(f"Geocoded with simplified address: {simple_addr}")
                else:
                    print(f"Could not geocode (even after simplification): {addr['address']} | Simplified: {simple_addr}")
            else:
                print(f"Could not geocode: {addr['address']}")

def main():
    all_form13s = read_form13_csv()
    if not all_form13s:
        print("No Form 13 data found.")
        return

    # expand_with_address_nodes will extract, geocode, and expand the graph
    # expand_with_address_nodes(kg, all_form13s)
    kg.refresh_schema
    print(textwrap.fill(kg.schema, 60))

if __name__ == "__main__":
    main()