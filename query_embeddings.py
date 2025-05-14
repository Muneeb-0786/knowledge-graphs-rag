from dotenv import load_dotenv
import os
import textwrap

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain

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

def get_neo4j_graph(cfg):
    return Neo4jGraph(
        url=cfg["NEO4J_URI"],
        username=cfg["NEO4J_USERNAME"],
        password=cfg["NEO4J_PASSWORD"],
        database=cfg["NEO4J_DATABASE"]
    )

def get_cypher_chain(kg, schema):
    CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:

# What investment firms are in San Francisco?
MATCH (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
    WHERE mgrAddress.city = 'San Francisco'
RETURN mgr.managerName

# What investment firms are near Santa Clara?
  MATCH (address:Address)
    WHERE address.city = "Santa Clara"
  MATCH (mgr:Manager)-[:LOCATED_AT]->(managerAddress:Address)
    WHERE point.distance(address.location, 
        managerAddress.location) < 10000
  RETURN mgr.managerName, mgr.managerAddress

# What does Palo Alto Networks do?
  CALL db.index.fulltext.queryNodes(
         "fullTextCompanyNames", 
         "Palo Alto Networks"
         ) YIELD node, score
  WITH node as com
  MATCH (com)-[:FILED]->(f:Form),
    (f)-[s:SECTION]->(c:Chunk)
  WHERE s.f10kItem = "item1"
RETURN c.text

The question is:
{question}"""

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], 
        template=CYPHER_GENERATION_TEMPLATE
    )

    return GraphCypherQAChain.from_llm(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0),
        graph=kg,
        verbose=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        allow_dangerous_requests=True,
    )

def chat_loop(cypher_chain, schema):
    print("Cypher Chat. Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        response = cypher_chain({"schema": schema, "query": question})  # changed 'question' to 'query'
        print("\nCypher statement:\n")
        print(response['result'])

def main():
    cfg = load_env_and_constants()
    kg = get_neo4j_graph(cfg)
    # You may want to fetch or define your schema string here
    schema = kg.schema  # Changed from kg.get_schema() to kg.schema
    cypher_chain = get_cypher_chain(kg, schema)
    chat_loop(cypher_chain, schema)
if __name__ == "__main__":
    main()
   