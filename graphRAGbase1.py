import networkx as nx
import matplotlib.pyplot as plt
from gemini_api import GeminiAPI


def generate_graph_from_document(document_path):
    """
    Generate a graph from a document using the Gemini API and NetworkX.

    Parameters:
    document_path (str): Path to the document to be analyzed.

    Returns:
    networkx.Graph: The generated graph.
    """
    # Initialize the Gemini API
    gemini = GeminiAPI()

    # Extract entities and relations from the document
    entities, relations = gemini.extract_entities_and_relations(document_path)

    # Create a NetworkX graph
    graph = nx.Graph()

    # Add entities as nodes
    for entity in entities:
        graph.add_node(entity, type=gemini.get_entity_type(entity))

    # Add relations as edges
    for relation in relations:
        graph.add_edge(relation.subject, relation.object, type=relation.type)

    return graph




def visualize_graph(graph, layout_algorithm='spring', output_file=None):
    """
    Visualize the generated graph using NetworkX and Matplotlib.

    Parameters:
    graph (networkx.Graph): The graph to be visualized.
    layout_algorithm (str, optional): The graph layout algorithm to use. Default is 'spring'.
    output_file (str, optional): If provided, the graph will be saved to the specified file.
    """
    # Choose the layout algorithm
    if layout_algorithm == 'spring':
        pos = nx.spring_layout(graph)
    elif layout_algorithm == 'circular':
        pos = nx.circular_layout(graph)
    elif layout_algorithm == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        raise ValueError("Invalid layout algorithm. Choose 'spring', 'circular', or 'kamada_kawai'.")

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_color=[graph.nodes[node]['type'] for node in graph.nodes])
    nx.draw_networkx_edges(graph, pos, edge_color=[graph.edges[edge]['type'] for edge in graph.edges])

    # Add a legend
    node_types = set([graph.nodes[node]['type'] for node in graph.nodes])
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type, markerfacecolor='C{}'.format(i), markersize=10) for i, node_type in enumerate(node_types)]
    plt.legend(handles=legend_elements, loc='upper left')

    # Show or save the plot
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()



def answer_question(graph, question):
    """
    Answer a question based on the information in the graph.

    Parameters:
    graph (networkx.Graph): The graph containing the document information.
    question (str): The question to be answered.

    Returns:
    str: The answer to the question.
    """
    # Implement your question answering logic here
    # You can use NetworkX functions to traverse the graph and extract relevant information
    # For example, you could use graph.neighbors() to find related entities
    
    # Sample implementation
    if question.startswith("What is the type of"):
        entity = question.split()[-1]
        if entity in graph.nodes:
            return f"The type of {entity} is {graph.nodes[entity]['type']}"
        else:
            return f"I could not find an entity named {entity} in the graph."
    else:
        return "I'm sorry, I don't understand the question. Please rephrase it or ask something else."



# Example usage
document_path = "path/to/your/document.txt"
graph = generate_graph_from_document(document_path)
visualize_graph(graph, layout_algorithm='spring', output_file='graph.png')


question = "What is the type of Barack Obama?"
answer = answer_question(graph, question)
print(answer)

