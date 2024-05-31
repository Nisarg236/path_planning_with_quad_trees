import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from itertools import count
import numpy as np

class Square:
    _ids = count(1)  # Counter for generating unique IDs
    def __init__(self, top_left, bottom_right):
        self.id = next(self._ids)
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.length = bottom_right[1] - top_left[1]
        self.width = bottom_right[0] - top_left[0]
        self.center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def to_dict(self):
        return {
            'id': self.id,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'length': self.length,
            'width': self.width,
            'center': self.center,
            'neighbors': [n.id for n in self.neighbors]
        }


# Function to check if a square area is free space
def is_free_space(grid, top_left, bottom_right, free_space_threshold = 240):
    """
    Checks if the roi defined between top_left and bottom_right coordinates is a free space.

    Parameters:
    - grid: 2D array representing the grid.
    - top_left: Tuple (x, y) representing the top-left corner of the subgrid.
    - bottom_right: Tuple (x, y) representing the bottom-right corner of the subgrid.

    Returns:
    - Boolean: True if the subgrid is all free space, False otherwise.
    """
    sub_grid = grid[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return np.all(sub_grid > free_space_threshold)


# Function to recursively partition the grid
def partition_grid(grid, x, y, size, square_size_threshold = 5):
    """
    Recursively divides the grid into four parts. If a part contains obstacles it further divied into four more parts.

    Parameters:
    - grid: 2D array representing the grid.
    - x, y: Top-left coordinates of the current partition.
    - size: Size of the current partition.

    Returns:
    - List of Square objects representing free spaces in the grid.
    """
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

# Function to load the graph from a JSON file
def load_graph_from_json(json_file):
    """
    Loads the saved quad-tree graph JSON file and constructs a NetworkX graph.

    Parameters:
    - json_file: Path to the JSON file containing the graph data.

    Returns:
    - G: A NetworkX graph with Square nodes.
    """
    with open(json_file, 'r') as file:
        graph_data = json.load(file)
    G = nx.Graph()
    for node_data in graph_data['nodes']:
        G.add_node(node_data['id'], center=tuple(node_data['center']))  # Ensure center is a tuple
        for neighbor_id in node_data['neighbors']:
            G.add_edge(node_data['id'], neighbor_id)
    return G
    

def bresenham_line(x1, y1, x2, y2):
    """Bresenham's line algorithm."""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

def quad_tree_to_graph(grid_squares, visualization):
    """
    Create a graph from grid squares and visualize the squares and their connections.

    Parameters:
    - grid_squares: List of grid square objects, each having `center`, `top_left`, `bottom_right`, and `add_neighbor` method.
    - visualization: Image on which the visualization is drawn.

    Returns:
    - G: A NetworkX graph with squares as nodes and their connections as edges.
    """
    G = nx.Graph()

    for square in grid_squares:
        # Add node to the graph
        G.add_node(square)
        
        # Draw a green circle at the center of the square
        cv2.circle(visualization, (square.center[1], square.center[0]), 0, (0, 255, 0), -1)

        # Draw the square in red
        cv2.rectangle(visualization, square.top_left[::-1], square.bottom_right[::-1], (0, 0, 255), 1)

        # Find and add neighbors
        for other_square in grid_squares:
            if square != other_square:
                # Check if the squares are vertical neighbors
                if (square.top_left[0] == other_square.bottom_right[0] or square.bottom_right[0] == other_square.top_left[0]) and \
                   (square.top_left[1] < other_square.bottom_right[1] and square.bottom_right[1] > other_square.top_left[1]):
                    square.add_neighbor(other_square)
                    G.add_edge(square, other_square)

                # Check if the squares are horizontal neighbors
                elif (square.top_left[1] == other_square.bottom_right[1] or square.bottom_right[1] == other_square.top_left[1]) and \
                     (square.top_left[0] < other_square.bottom_right[0] and square.bottom_right[0] > other_square.top_left[0]):
                    square.add_neighbor(other_square)
                    G.add_edge(square, other_square)

    return G

def euclidean(p1, p2):
    """
    Calculates the Euclidean distance between two points using NumPy.
    
    Parameters:
    - p1: A tuple (x, y) representing the first point.
    - p2: A tuple (x, y) representing the second point.
    
    Returns:
    - distance: The Euclidean distance between the two points.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def high_level_path_to_pixels(high_level_path, graph):
    """
    Converts a high-level path to a detailed pixel-level path. Traces a line between nodes and returns all pixels under the line.
    
    Parameters:
    - high_level_path: List of nodes representing the high-level path.
    - graph: A NetworkX graph where nodes have a 'center' attribute (tuple of (x, y) coordinates).

    Returns:
    - pixel_path: List of (x, y) tuples representing the pixel-level path.
    """
    pixel_path = []
    
    for i in range(len(high_level_path) - 1):
        node1, node2 = high_level_path[i], high_level_path[i + 1]
        x1, y1 = graph.nodes[node1]['center']
        x2, y2 = graph.nodes[node2]['center']
        pixels_between_nodes = bresenham_line(x1, y1, x2, y2)
        pixel_path.extend(pixels_between_nodes)
    return pixel_path


def find_nearest_high_level_node(graph, pixel):
    """
    Finds the nearest high-level node in the graph to a given pixel.
    
    Parameters:
    - graph: A NetworkX graph where nodes have a 'center' attribute (tuple of (x, y) coordinates).
    - pixel: A tuple (x, y) representing the pixel coordinates.
    
    Returns:
    - nearest_node: The node in the graph that is closest to the given pixel.
    """
    min_distance = float('inf')
    nearest_node = None
    # Iterate over all nodes and their data in the graph
    for node, data in graph.nodes(data=True):
        # Calculate the Euclidean distance between the pixel and the node's center
        distance = euclidean(pixel, data['center'])
        # Update the nearest node if the current distance is smaller
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node


def find_high_level_path(G, start_pixel, end_pixel):
    """
    Finds a high-level path in the graph from a start pixel to an end pixel.
    
    Parameters:
    - G: A NetworkX graph where nodes have a 'center' attribute (tuple of (x, y) coordinates).
    - start_pixel: A tuple (x, y) representing the start pixel coordinates.
    - end_pixel: A tuple (x, y) representing the end pixel coordinates.
    
    Returns:
    - path: A list of nodes representing the high-level path from the start node to the end node.
    """
    # Find the nearest high-level node to the start pixel
    start_node = find_nearest_high_level_node(G, start_pixel)
    # Find the nearest high-level node to the end pixel
    end_node = find_nearest_high_level_node(G, end_pixel)
    # Use A* algorithm to find the shortest path between the start and end nodes
    return nx.astar_path(G, start_node, end_node, weight="cost")
