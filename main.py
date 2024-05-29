from grid_to_graph import partition_grid, quad_tree_to_graph
import cv2
import matplotlib.pyplot as plt
import json

image_path = "map.png"
occupancy_grid = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define parameters
free_space_threshold = 240  # Assume any pixel value greater than this is free space
square_size_threshold = 5  # Minimum size of a square

grid_squares = partition_grid(occupancy_grid, 0, 0, occupancy_grid.shape[0])

# Create an RGB image for visualization
visualization = cv2.cvtColor(occupancy_grid, cv2.COLOR_GRAY2RGB)

# Create a graph
G = quad_tree_to_graph(grid_squares, visualization)

for edge in G.edges():
    start_point = edge[0].center
    end_point = edge[1].center
    plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 'b-', linewidth=1)

# Convert the graph to a JSON-friendly format
graph_data = {
    'nodes': [square.to_dict() for square in G.nodes]
}

# Write the graph to a JSON file
with open('graph.json', 'w') as file:
    json.dump(graph_data, file)

# Display the image with nodes, edges, and grid squares
plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()