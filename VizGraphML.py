import networkx as nx
import json
from google.colab import files
from IPython.display import HTML, IFrame, display
import base64
import http.server
import socketserver
import threading
import os
import subprocess

# Install required packages
!pip install networkx pyngrok
from pyngrok import ngrok

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def graphml_to_json(graphml_file):
    """Convert GraphML to JSON format"""
    G = nx.read_graphml(graphml_file)
    data = nx.node_link_data(G)
    return json.dumps(data)

def save_visualization(json_data, output_dir="graph_viz"):
    """Save visualization files to disk"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save JSON data
    json_path = os.path.join(output_dir, 'graph_data.js')
    with open(json_path, 'w') as f:
        f.write(f"const graphJson = {json_data};")
    
    # Save HTML file
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Graph Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
            }}
            svg {{
                width: 100%;
                height: 100%;
            }}
            .links line {{
                stroke: #999;
                stroke-opacity: 0.6;
            }}
            .nodes circle {{
                stroke: #fff;
                stroke-width: 1.5px;
            }}
            .node-label {{
                font-size: 12px;
                pointer-events: none;
            }}
            .link-label {{
                font-size: 10px;
                fill: #666;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s;
            }}
            .link:hover .link-label {{
                opacity: 1;
            }}
            .tooltip {{
                position: absolute;
                text-align: left;
                padding: 10px;
                font: 12px sans-serif;
                background: lightsteelblue;
                border: 0px;
                border-radius: 8px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s;
                max-width: 300px;
            }}
            .legend {{
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
            }}
            .legend-item {{
                margin: 5px 0;
            }}
            .legend-color {{
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 5px;
                vertical-align: middle;
            }}
        </style>
    </head>
    <body>
        <svg></svg>
        <div class="tooltip"></div>
        <div class="legend"></div>
        <script src="graph_data.js"></script>
        <script>
            const svg = d3.select("svg"),
                width = window.innerWidth,
                height = window.innerHeight;

            svg.attr("viewBox", [0, 0, width, height]);

            const g = svg.append("g");

            const entityTypes = [...new Set(graphJson.nodes.map(d => d.entity_type))];
            const color = d3.scaleOrdinal(d3.schemeCategory10).domain(entityTypes);

            const simulation = d3.forceSimulation(graphJson.nodes)
                .force("link", d3.forceLink(graphJson.links).id(d => d.id).distance(150))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(30));

            const linkGroup = g.append("g")
                .attr("class", "links")
                .selectAll("g")
                .data(graphJson.links)
                .enter().append("g")
                .attr("class", "link");

            const link = linkGroup.append("line")
                .attr("stroke-width", d => Math.sqrt(d.value || 1));

            const linkLabel = linkGroup.append("text")
                .attr("class", "link-label")
                .text(d => d.description || "");

            const node = g.append("g")
                .attr("class", "nodes")
                .selectAll("circle")
                .data(graphJson.nodes)
                .enter().append("circle")
                .attr("r", 5)
                .attr("fill", d => color(d.entity_type))
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            const nodeLabel = g.append("g")
                .attr("class", "node-labels")
                .selectAll("text")
                .data(graphJson.nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .text(d => d.id);

            const tooltip = d3.select(".tooltip");

            node.on("mouseover", function(event, d) {{
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`<strong>${{d.id}}</strong><br>Entity Type: ${{d.entity_type}}<br>Description: ${{d.description || "N/A"}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", function(d) {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }});

            const legend = d3.select(".legend");
            entityTypes.forEach(type => {{
                legend.append("div")
                    .attr("class", "legend-item")
                    .html(`<span class="legend-color" style="background-color: ${{color(type)}}"></span>${{type}}`);
            }});

            simulation
                .nodes(graphJson.nodes)
                .on("tick", ticked);

            simulation.force("link")
                .links(graphJson.links);

            function ticked() {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                linkLabel
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2)
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle");

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                nodeLabel
                    .attr("x", d => d.x + 8)
                    .attr("y", d => d.y + 3);
            }}

            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }}

            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}

            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }}

            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", zoomed);

            svg.call(zoom);

            function zoomed(event) {{
                g.attr("transform", event.transform);
            }}
        </script>
    </body>
    </html>
    '''
    
    html_path = os.path.join(output_dir, 'index.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return output_dir

def start_server(directory, port=8000):
    """Start HTTP server"""
    os.chdir(directory)
    handler = CustomHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()

def start_ngrok(port):
    """Start ngrok tunnel"""
    public_url = ngrok.connect(port)
    print(f"Public URL: {public_url}")
    return public_url

def visualize_graphml(graphml_file, save=False, serve=False, external_access=False, port=8000):
    """Main function to visualize GraphML file with multiple deployment options"""
    # Convert GraphML to JSON
    json_data = graphml_to_json(graphml_file)
    
    if save or serve:
        # Save files to disk
        output_dir = save_visualization(json_data)
        print(f"Files saved to {output_dir}")
        
        if serve:
            # Start local server in a separate thread
            server_thread = threading.Thread(target=start_server, args=(output_dir, port))
            server_thread.daemon = True
            server_thread.start()
            print(f"Local server started at http://localhost:{port}")
            
            if external_access:
                # Start ngrok for external access
                public_url = start_ngrok(port)
                print(f"External access URL: {public_url}")
    else:
        # Display directly in Colab
        display(HTML(create_visualization(json_data)))

# Example usage
print("Please upload your GraphML file...")
uploaded = files.upload()
graphml_file = next(iter(uploaded))

# Choose your preferred deployment method:

# 1. Display directly in Colab
visualize_graphml(graphml_file)

# 2. Save files locally and serve through HTTP
# visualize_graphml(graphml_file, save=True, serve=True, port=8000)

# 3. Save files, serve through HTTP, and enable external access through ngrok
# visualize_graphml(graphml_file, save=True, serve=True, external_access=True, port=8000)
