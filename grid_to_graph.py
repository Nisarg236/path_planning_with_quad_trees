import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from itertools import count
from square import Square

# Load the occupancy grid PNG using OpenCV
image_path = "map.png"
occupancy_grid = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define parameters
free_space_threshold = 240  # Assume any pixel value greater than this is free space
square_size_threshold = 5  # Minimum size of a square

# Function to check if a square area is free space
def is_free_space(grid, top_left, bottom_right):
    sub_grid = grid[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return np.all(sub_grid > free_space_threshold)

# Function to recursively partition the grid
def partition_grid(grid, x, y, size):
    if size < square_size_threshold:
        top_left = (x, y)
        bottom_right = (x + size, y + size)
        if is_free_space(grid, top_left, bottom_right):
            return [Square(top_left, bottom_right)]
        else:
            return []
    else:
        # Check if the area is homogeneous and free space
        if is_free_space(grid, (x, y), (x + size, y + size)):
            top_left = (x, y)
            bottom_right = (x + size, y + size)
            return [Square(top_left, bottom_right)]
        else:
            # Divide the area into quadrants and partition recursively
            new_size = size // 2
            quad1 = partition_grid(grid, x, y, new_size)
            quad2 = partition_grid(grid, x, y + new_size, new_size)
            quad3 = partition_grid(grid, x + new_size, y, new_size)
            quad4 = partition_grid(grid, x + new_size, y + new_size, new_size)
            return quad1 + quad2 + quad3 + quad4

# Partition the grid
grid_squares = partition_grid(occupancy_grid, 0, 0, occupancy_grid.shape[0])

# Create an RGB image for visualization
visualization = cv2.cvtColor(occupancy_grid, cv2.COLOR_GRAY2RGB)

# Create a graph
G = nx.Graph()

# Add nodes and edges to the graph
for square in grid_squares:
    G.add_node(square)
    cv2.circle(visualization, (square.center[1], square.center[0]), 0, (0, 255, 0), -1)  # Draw a green node at the center

    # Draw the square in red
    cv2.rectangle(visualization, square.top_left[::-1], square.bottom_right[::-1], (0, 0, 255), 1)

    # Find and add neighbors
    for other_square in grid_squares:
        if square != other_square:
            if (square.top_left[0] == other_square.bottom_right[0] or square.bottom_right[0] == other_square.top_left[0]) and \
               (square.top_left[1] < other_square.bottom_right[1] and square.bottom_right[1] > other_square.top_left[1]):
                square.add_neighbor(other_square)
                G.add_edge(square, other_square)

            elif (square.top_left[1] == other_square.bottom_right[1] or square.bottom_right[1] == other_square.top_left[1]) and \
                 (square.top_left[0] < other_square.bottom_right[0] and square.bottom_right[0] > other_square.top_left[0]):
                square.add_neighbor(other_square)
                G.add_edge(square, other_square)
                     
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
