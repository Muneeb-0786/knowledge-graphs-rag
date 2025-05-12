from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env file

# Load variables from .env file
load_dotenv()

# Connection details from environment variables
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database_name = os.getenv("NEO4J_DATABASE")

# Create driver instance
driver = GraphDatabase.driver(uri, auth=(username, password))

# Query function
def get_movie_count():
    with driver.session() as session:
        cypher = """
    MATCH (n:Person)
    WHERE n.name = "Keanu Reeves"
RETURN n.name AS name
UNION
MATCH (n:Movie)
RETURN n.title AS name
  """
        result = session.run(cypher)
        record = result.single()
        print("name:", record["name"])

query = """
CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
FOR (m:Movie) ON (m.taglineEmbedding) 
OPTIONS { indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}"""


def create_movies_embeddings():
    with driver.session() as session:
        session.run(query)
        print("Index created successfully.")
        close_driver()
        

def close_driver():
    driver.close()
    print("Driver closed successfully.")

        
def query_embeddings():
    with driver.session() as session:
        # First, get all movies that need embeddings
        fetch_cypher = """
        MATCH (movie:Movie)
        WHERE movie.tagline IS NOT NULL AND movie.taglineEmbedding IS NULL
        RETURN movie.tagline as tagline, ID(movie) as movieId
        """
        movies = list(session.run(fetch_cypher))
        
        # Process each movie
        updated_count = 0
        for movie in movies:
            tagline = movie["tagline"]
            movie_id = movie["movieId"]
            
            # Generate embedding using Google API
            embedding = generate_embedding(tagline)
            
            if embedding:
                # Update the movie with the embedding
                update_cypher = """
                MATCH (movie:Movie)
                WHERE ID(movie) = $movieId
                SET movie.taglineEmbedding = $embedding
                """
                session.run(update_cypher, {"movieId": movie_id, "embedding": embedding})
                updated_count += 1
        
        print(f"Vector Embeddings created: {updated_count}")
        close_driver()

def generate_embedding(text):
    """Generate vector embedding for text using Google's Generative AI API"""
    api_key = os.getenv("GEMINI_API_KEY")
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "model": "models/embedding-001",
        "content": {
            "parts": [
                {"text": text}
            ]
        }
    }

    url = f"{endpoint}?key={api_key}"

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        
        # Extract embeddings with proper validation
        if "embedding" in result and "values" in result["embedding"]:
            embedding_values = result["embedding"]["values"]
            print(f"Generated embedding with {len(embedding_values)} dimensions")
            return embedding_values
        else:
            print(f"Unexpected API response structure: {result}")
            return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        if 'response' in locals():
            print(f"Response content: {response.text}")
        return None

def get_movie_embeddings():
    question = "What movies are about adventure?"
    
    # Generate embedding for the question
    question_embedding = generate_embedding(question)
    
    if not question_embedding:
        print("Failed to generate embedding for the question")
        return
    
    print(f"Embedding length: {len(question_embedding)}")
    
    # First, recreate the index to ensure it has the right dimensions
    with driver.session() as session:
        try:
            # Drop existing index if it exists
            session.run("DROP INDEX movie_tagline_embeddings IF EXISTS")
            print("Dropped existing index")
            
            # Create index with correct dimensions
            session.run(query)
            print("Created new index with correct dimensions")
        except Exception as e:
            print(f"Error recreating index: {e}")
    
    with driver.session() as session:
        cypher = """
        CALL db.index.vector.queryNodes(
            'movie_tagline_embeddings', 
            $top_k, 
            $question_embedding
        ) YIELD node AS movie, score
        RETURN movie.title AS title, movie.tagline AS tagline, score
        """
        params = {
            "question_embedding": question_embedding,
            "top_k": 5
        }

        result = session.run(cypher, params)
        print("Top matching movies:")
        for record in result:
            print(f"{record['title']} ({record['score']:.4f}) — {record['tagline']}")

    close_driver()

def clear_all_embeddings():
    """Remove all taglineEmbedding properties from Movie nodes"""
    with driver.session() as session:
        clear_cypher = """
        MATCH (movie:Movie)
        WHERE movie.taglineEmbedding IS NOT NULL
        REMOVE movie.taglineEmbedding
        """
        result = session.run(clear_cypher)
        count = result.consume().counters.properties_set
        print(f"Cleared embeddings from {count} movies")

# To fix the issue, first recreate the index with the correct dimensions,
# then generate embeddings for the movies, and finally query them
def main():
    # First clear all existing embeddings
    clear_all_embeddings()
    
    # Then create/recreate the index with correct dimensions
    with driver.session() as session:
        try:
            session.run("DROP INDEX movie_tagline_embeddings IF EXISTS")
            session.run(query)
            print("Index created with 768 dimensions")
        except Exception as e:
            print(f"Error creating index: {e}")
    
    # Then generate embeddings for movies
    query_embeddings()
    
   
 # Now query the embeddings
get_movie_embeddings()
