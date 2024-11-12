import argparse
import networkx as nx
import matplotlib.pyplot as plt
from openai import OpenAI
from collections import defaultdict
from spacy.lang.en import English

# Initialize OpenAI client (you'll need to set your API key)
client = OpenAI()

# Initialize spaCy English model for coreference resolution
nlp = English()
nlp.add_pipe("neuralcoref")

def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

def extract_elements_from_chunks(chunks):
    elements = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract key entities, relationships, and relationship types from the following text:"},
                {"role": "user", "content": chunk}
            ]
        )
        elements.append(response.choices[0].message.content)
    return elements

def resolve_coreferences(text):
    doc = nlp(text)
    return [str(mention) for cluster in doc._.coref_clusters for mention in cluster]

def build_graph_from_summaries(summaries):
    G = nx.Graph()
    entity_relationships = defaultdict(list)

    for summary in summaries:
        lines = summary.split("\n")
        entities = []
        for line in lines:
            if line.startswith("Entity:"):
                entity = line.split(":")[1].strip()
                entities.append(entity)
                G.add_node(entity)
            elif line.startswith("Relationship:"):
                parts = line.split(":")[1].strip().split(" - ")
                if len(parts) == 3:
                    subject, relation, object = parts
                    entity_relationships[subject].append((relation, object))
                    G.add_edge(subject, object, label=relation)

    # Resolve coreferences and update the graph
    for entity, relationships in entity_relationships.items():
        resolved_entity = resolve_coreferences(entity)[0]
        if resolved_entity != entity:
            G.add_node(resolved_entity)
            for relation, obj in relationships:
                G.add_edge(resolved_entity, obj, label=relation)
            G.remove_node(entity)

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
    graph = build_graph_from_summaries(elements)
    visualize_graph(graph, output_file=args.output_file)

if __name__ == "__main__":
    main()

    
