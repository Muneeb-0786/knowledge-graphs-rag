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

# Global constants for Neo4j vector search
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'



# Neo4j connection management
def get_neo4j_driver():
    """Create and return a Neo4j driver connection"""
    return GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        database=NEO4J_DATABASE
    )

def execute_neo4j_query(query, params=None):
    """Execute a Neo4j query and return the results"""
    driver = get_neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, params)
            return list(result)
    finally:
        driver.close()

# Text processing functions
def split_form10k_data_from_file(file_path, max_chunks_per_item=20):
    """Split a Form 10-K JSON file into chunks with metadata"""
    chunks_with_metadata = []
    
    try:
        with open(file_path) as f:
            file_as_object = json.load(f)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Process each relevant section of the 10-K
        for item in ['item1', 'item1a', 'item7', 'item7a']:
            print(f'Processing {item} from {file_path}')
            
            item_text = file_as_object.get(item, '')
            if not item_text:
                print(f"\tSkipping {item}: No content found")
                continue
                
            item_text_chunks = text_splitter.split_text(item_text)
            form_id = file_path[file_path.rindex('/') + 1:file_path.rindex('.')]            
            # Create chunks with metadata
            for chunk_seq_id, chunk in enumerate(item_text_chunks[:max_chunks_per_item]):
                chunks_with_metadata.append({
                    'text': chunk, 
                    'f10kItem': item,
                    'chunkSeqId': chunk_seq_id,
                    'formId': form_id,
                    'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                    'names': file_as_object.get('names', ''),
                    'cik': file_as_object.get('cik', ''),
                    'cusip6': file_as_object.get('cusip6', ''),
                    'source': file_as_object.get('source', ''),
                })
            
            print(f'\tSplit into {min(len(item_text_chunks), max_chunks_per_item)} chunks')
            
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        
    return chunks_with_metadata

# Embedding generation
def generate_embeddings(text):
    """Generate embeddings for text using Google's Generative AI API"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    
    try:
        response = requests.post(url, headers=headers, params=params, json=data)
        if response.status_code == 200:
            embedding_data = response.json()
            return embedding_data.get("embedding", {}).get("values", [])
        else:
            print(f"Error generating embeddings (Status {response.status_code}): {response.text}")
            return []
    except Exception as e:
        print(f"Exception while generating embeddings: {str(e)}")
        return []

# Neo4j database setup and management
def setup_neo4j_constraints_and_indexes():
    """Set up required constraints and indexes in Neo4j"""
    # Create constraint for chunk uniqueness
    try:
        constraint_query = """
        CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
            FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
        """
        execute_neo4j_query(constraint_query)
        print("Created or confirmed uniqueness constraint for Chunk.chunkId")
    except Exception as e:
        print(f"Error creating uniqueness constraint: {str(e)}")

    # Create vector index
    try:
        vector_index_query = f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:{VECTOR_NODE_LABEL})
        ON (c.{VECTOR_EMBEDDING_PROPERTY})
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: 768,
          `vector.similarity_function`: 'cosine'
        }}}}
        """
        execute_neo4j_query(vector_index_query)
        print(f"Created or confirmed vector index: {VECTOR_INDEX_NAME}")
    except Exception as e:
        print(f"Error creating vector index: {str(e)}")

def load_chunks_to_neo4j(chunks):
    """Load chunks with embeddings into Neo4j database"""
    # Check if chunk exists with embedding
    check_chunk_exists_query = """
    MATCH (c:Chunk {chunkId: $chunkId})
    WHERE c.textEmbedding IS NOT NULL
    RETURN count(c) > 0 as exists
    """
    
    # Define the merge query
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
            mergedChunk.text = $chunkParam.text,
            mergedChunk.textEmbedding = $embedding
    RETURN mergedChunk
    """
    
    loaded_count = 0
    skipped_count = 0
    
    driver = get_neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for chunk in chunks:
                try:
                    # Check if chunk with embedding already exists
                    result = session.run(
                        check_chunk_exists_query, 
                        {"chunkId": chunk['chunkId']}
                    ).single()
                    
                    # If the chunk exists with embedding, skip it
                    if result and result["exists"]:
                        print(f"Skipping chunk {chunk['chunkId']} - embedding already exists")
                        skipped_count += 1
                        continue
                        
                    # Generate embedding for the chunk text
                    embedding = generate_embeddings(chunk['text'])
                    if embedding:
                        # Execute the merge query with chunk data and embedding
                        session.run(
                            merge_chunk_node_query,
                            {"chunkParam": chunk, "embedding": embedding}
                        )
                        loaded_count += 1
                        print(f"Loaded chunk {chunk['chunkId']} with embedding")
                except Exception as e:
                    print(f"Error processing chunk {chunk['chunkId']}: {str(e)}")
    finally:
        driver.close()
    
    print(f"Total chunks loaded: {loaded_count}, skipped (already existed): {skipped_count}")
    return loaded_count

def add_embedding_vectors():
    """Add embeddings to chunks that don't have them yet"""
    driver = get_neo4j_driver()
    count = 0
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # Find chunks without embeddings
            find_chunks_query = """
                MATCH (chunk:Chunk) 
                WHERE chunk.textEmbedding IS NULL
                RETURN chunk.chunkId as chunkId, chunk.text as text
                LIMIT 100
            """
            chunks_to_process = session.run(find_chunks_query).data()
            
            for chunk in chunks_to_process:
                embedding = generate_embeddings(chunk['text'])
                
                if embedding:
                    # Update the chunk with the embedding
                    update_query = """
                        MATCH (c:Chunk {chunkId: $chunkId})
                        SET c.textEmbedding = $embedding
                        RETURN c
                    """
                    session.run(update_query, {"chunkId": chunk['chunkId'], "embedding": embedding})
                    count += 1
                    print(f"Added embedding to chunk {chunk['chunkId']}")
            
            print(f"Added embedding vectors to {count} chunks")
    finally:
        driver.close()

def clean_duplicate_chunks(force=False):
    """Clean up duplicate chunk nodes in Neo4j"""
    if not force:
        print("\nTo clean up duplicate chunks, run with force=True parameter")
        return
        
    cleanup_query = """
    MATCH (c:Chunk)
    WITH c.chunkId as chunkId, collect(c) as duplicates
    WHERE size(duplicates) > 1
    UNWIND tail(duplicates) as dupe
    DETACH DELETE dupe
    RETURN count(*) as deletedNodes
    """
    result = execute_neo4j_query(cleanup_query)
    print(f"Deleted {result[0]['deletedNodes']} duplicate nodes")

def check_chunk_statistics():
    """Check statistics about chunks in the Neo4j database"""
    print("\n--- Database Statistics ---")
    
    # Count all nodes
    all_nodes_query = "MATCH (n) RETURN count(n) as count"
    result = execute_neo4j_query(all_nodes_query)
    print(f"Total nodes in database: {result[0]['count']}")
    
    # Count Chunk nodes
    chunk_nodes_query = "MATCH (c:Chunk) RETURN count(c) as count"
    result = execute_neo4j_query(chunk_nodes_query)
    print(f"Total Chunk nodes: {result[0]['count']}")
    
    # Count Chunk nodes with embeddings
    chunk_with_embedding_query = "MATCH (c:Chunk) WHERE c.textEmbedding IS NOT NULL RETURN count(c) as count"
    result = execute_neo4j_query(chunk_with_embedding_query)
    print(f"Chunk nodes with embeddings: {result[0]['count']}")
    
    # Check for duplicate chunkIds
    duplicate_check_query = """
    MATCH (c:Chunk)
    WITH c.chunkId as chunkId, count(*) as count
    WHERE count > 1
    RETURN chunkId, count ORDER BY count DESC LIMIT 10
    """
    result = execute_neo4j_query(duplicate_check_query)
    if result:
        print("\nFound duplicate chunk IDs:")
        for record in result:
            print(f"  {record['chunkId']}: {record['count']} duplicates")
    else:
        print("\nNo duplicate chunk IDs found")

# Search and retrieval functions
def neo4j_vector_search(question, top_k=10):
    """Search for similar nodes using Neo4j vector search"""
    driver = get_neo4j_driver()
    try:
        # Generate embedding for the question
        question_embedding = generate_embeddings(question)
        if not question_embedding:
            return []
        
        # Use vector search query
        vector_search_query = """
          CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) 
          YIELD node, score
          RETURN score, node.text AS text
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                vector_search_query, 
                {
                    'embedding': question_embedding,
                    'index_name': VECTOR_INDEX_NAME, 
                    'top_k': top_k
                }
            )
            return result.data()
    finally:
        driver.close()

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

def answer_question(question, qa_chain=None):
    """Answer a question using the QA chain"""
    if not qa_chain:
        qa_chain = setup_langchain_qa()
        
    response = qa_chain({"question": question}, return_only_outputs=True)
    return response['answer']

def prettychain(question, qa_chain=None):
    """Pretty print the chain's response to a question"""
    answer = answer_question(question, qa_chain)
    print(textwrap.fill(answer, 60))

def create_form(form_info):
    driver = get_neo4j_driver()

    with driver.session(database=NEO4J_DATABASE) as session:
        exist = session.run("MATCH (f:Form) RETURN count(f) as formCount").single()
        print(f"Total Form nodes: {exist['formCount']}")
        if exist['formCount'] > 0:
            print("Form node already exists")
            return

        """Create a Form node in Neo4j"""
        cypher = """
        MERGE (f:Form {formId: $formInfoParam.formId })
        ON CREATE 
            SET f.names = $formInfoParam.names,
            f.source = $formInfoParam.source,
            f.cik = $formInfoParam.cik,
            f.cusip6 = $formInfoParam.cusip6
        """
        driver = get_neo4j_driver()
        try:
            with driver.session(database=NEO4J_DATABASE) as session:
                formInfo = session.run(cypher, formInfoParam=form_info)
                if formInfo:
                    print(f"Form node created with info: {form_info}")
                    result = session.run("MATCH (f:Form) RETURN count(f) as formCount").single()
                    print(f"Total Form nodes: {result['formCount']}")
                else:
                    print("No Form node created")
        finally:
            driver.close()

def create_form10k_node():
    driver = get_neo4j_driver()
    cypher = """
  MATCH (anyChunk:Chunk) 
  WITH anyChunk LIMIT 1
  RETURN anyChunk { .names, .source, .formId, .cik, .cusip6 } as formInfo
"""
    form_info_list = driver.session().run(cypher).data()
    if form_info_list:
        form_info = form_info_list[0]['formInfo']

        print(f"Form 10-K node created with info: {form_info}")
        # Create the Form node
        # create_form(form_info)
        create_form(form_info)
    else:
        print("No Form 10-K node found")
    driver.close()

    
    
# Main execution flow
if __name__ == "__main__":
    # Initial file processing
    first_file_name = "./data/form10k/0000950170-23-027948.json"
    first_file_chunks = split_form10k_data_from_file(first_file_name)
    print(f'First file has {len(first_file_chunks)} chunks')
    
    # Set up Neo4j database
    setup_neo4j_constraints_and_indexes()
    
    # Load chunks to Neo4j
    print("Loading chunks to Neo4j with embeddings...")
    chunks_loaded = load_chunks_to_neo4j(first_file_chunks)
    print(f"Completed loading {chunks_loaded} chunks")
    
    # Check database statistics
    check_chunk_statistics()
    create_form10k_node()
    
    # # Example vector search
    # search_results = neo4j_vector_search('In a single sentence, tell me about Netapp.')
    # print(f"Found {len(search_results)} results")
    # print('--- Search Results ---')
    # for result in search_results:
    #     print(f"Score: {result['score']:.4f} - Text: {result['text'][:100]}...")
    # print('--- End of Results ---')
    
    # # Set up QA chain and answer a question
    # qa_chain = setup_langchain_qa()
    # question = "Where is Netapp headquartered?"
    # prettychain(question, qa_chain)
