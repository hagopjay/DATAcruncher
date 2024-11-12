import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
import openai
from chromadb.api import ChromaDB
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Tuple

# Initialize OpenAI client (you'll need to set your API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB client
chroma_client = ChromaDB(
    Settings(
        chroma_db_impl="deta",
        persist_directory="./chromadb",
    )
)
collection = chroma_client.create_collection(name="graphrag")

# Initialize Voyage API client (replace with your actual API key)
voyage_api_key = os.getenv("VOYAGE_API_KEY")
voyage_embedding_function = embedding_functions.get_function("text-embedding-ada-002", api_key=voyage_api_key)

def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100) -> List[str]:
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

def extract_elements_from_chunks(chunks) -> List[Tuple[List[str], List[Tuple[str, str, str]]]]:
    elements = []
    for chunk in chunks:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Extract key entities, relationships, and relationship types from the following text:\n\n{chunk}",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        text = response.choices[0].text.strip()
        entities = []
        relationships = []
        for line in text.split("\n"):
            if line.startswith("Entity:"):
                entity = line.split(":")[1].strip()
                entities.append(entity)
            elif line.startswith("Relationship:"):
                parts = line.split(":")[1].strip().split(" - ")
                if len(parts) == 3:
                    subject, relation, object = parts
                    relationships.append((subject, relation, object))
        elements.append((entities, relationships))
    return elements

def build_graph_from_summaries(elements) -> nx.Graph:
    G = nx.Graph()
    for entities, relationships in elements:
        for entity in entities:
            G.add_node(entity)
        for subject, relation, object in relationships:
            G.add_edge(subject, object, label=relation)
    return G

def visualize_graph(G, output_file=None):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title("GraphRAG Visualization")
    plt.axis('off')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def store_elements_in_chroma(elements):
    for entities, relationships in elements:
        for subject, relation, object in relationships:
            collection.add(
                documents=[subject, relation, object],
                ids=[f"{subject}-{relation}-{object}"],
                embeddings=[voyage_embedding_function(subject), voyage_embedding_function(relation), voyage_embedding_function(object)],
            )

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Visualization Tool")
    parser.add_argument("--input-file", type=str, required=True, help="Input file containing text data")
    parser.add_argument("--chunk-size", type=int, default=600, help="Size of text chunks for processing")
    parser.add_argument("--overlap-size", type=int, default=100, help="Overlap size between text chunks")
    parser.add_argument("--output-file", type=str, default=None, help="Output file for the graph visualization")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        documents = [f.read()]

    chunks = split_documents_into_chunks(documents, chunk_size=args.chunk_size, overlap_size=args.overlap_size)
    elements = extract_elements_from_chunks(chunks)
    store_elements_in_chroma(elements)
    graph = build_graph_from_summaries(elements)
    visualize_graph(graph, output_file=args.output_file)

if __name__ == "__main__":
    main()
