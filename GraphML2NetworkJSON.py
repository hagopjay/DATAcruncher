import xml.etree.ElementTree as ET
import json

def graphml_to_json(graphml_file):
    """
    Convert a GraphML file to JSON network format with nodes and links.
    
    Args:
        graphml_file (str): Path to the GraphML file
        
    Returns:
        dict: Dictionary containing nodes and links in the desired format
    """
    # Parse the GraphML file
    tree = ET.parse(graphml_file)
    root = tree.getroot()
    
    # GraphML uses namespaces, so we need to handle them
    # Remove the namespace prefix for easier processing
    namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    
    # Initialize output structure
    output = {
        "nodes": [],
        "links": []
    }
    
    # Dictionary to store node attributes
    node_attrs = {}
    
    # First, find all key definitions
    for key in root.findall('graphml:key', namespace):
        key_id = key.get('id')
        key_for = key.get('for')
        key_name = key.get('attr.name')
        if key_for == 'node':
            node_attrs[key_id] = key_name
    
    # Process all nodes
    graph = root.find('graphml:graph', namespace)
    if graph is not None:
        # Process nodes
        for node in graph.findall('graphml:node', namespace):
            node_id = node.get('id')
            node_data = {"id": node_id, "group": 1}  # Default group to 1
            
            # Process node attributes
            for data in node.findall('graphml:data', namespace):
                key = data.get('key')
                if key in node_attrs:
                    attr_name = node_attrs[key]
                    if attr_name == 'group':
                        try:
                            node_data['group'] = int(data.text)
                        except (ValueError, TypeError):
                            pass
            
            output["nodes"].append(node_data)
        
        # Process edges
        for edge in graph.findall('graphml:edge', namespace):
            source = edge.get('source')
            target = edge.get('target')
            
            # Default value to 1 if not specified
            value = 1
            for data in edge.findall('graphml:data', namespace):
                if data.get('key') == 'value':
                    try:
                        value = int(data.text)
                    except (ValueError, TypeError):
                        pass
            
            link = {
                "source": source,
                "target": target,
                "value": value
            }
            output["links"].append(link)
    
    return output

def save_json_network(data, output_file):
    """
    Save the network data to a JSON file.
    
    Args:
        data (dict): Network data dictionary
        output_file (str): Path to the output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Example GraphML file
    graphml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
        <key id="group" for="node" attr.name="group" attr.type="int"/>
        <key id="value" for="edge" attr.name="value" attr.type="int"/>
        <graph id="G" edgedefault="undirected">
            <node id="Myriel">
                <data key="group">1</data>
            </node>
            <node id="Napoleon">
                <data key="group">1</data>
            </node>
            <edge source="Napoleon" target="Myriel">
                <data key="value">1</data>
            </edge>
        </graph>
    </graphml>
    """
    
    # Save example GraphML to a file
    with open('example.graphml', 'w') as f:
        f.write(graphml_content)
    
    # Convert and save
    network_data = graphml_to_json('/content/nano-graphrag/LightRAG/dickens/graph_chunk_entity_relation.graphml')
    save_json_network(network_data, '/content/network.json')
