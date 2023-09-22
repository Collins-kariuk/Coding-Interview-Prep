### BREADTH-FIRST SEARCH ###
"""
Breadth-First Search (BFS) is an algorithm used to traverse or search through data structures like trees and graphs.
It explores all the vertices (or nodes) at the current level before moving on to the next level.
BFS is often used to find the shortest path between two nodes in an unweighted graph and can also
be used for various other tasks like finding connected components.

Step-by-step algorithm:

1. Create a queue (usually implemented using a list) to keep track of nodes to be explored.
2. Start from the initial node or vertex and enqueue it into the queue.
3. Mark the initial node as visited (to avoid revisiting it).
4. While the queue is not empty, do the following:
    a. Dequeue a node from the front of the queue.
    b. Process the dequeued node (e.g., print it or perform some operation).
    c. Enqueue all adjacent unvisited nodes of the dequeued node into the queue.
    d. Mark each adjacent node as visited.
"""

import collections
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs(self, start):
        # Create a queue for BFS
        queue = collections.deque()
        
        # Mark the start node as visited and enqueue it
        visited = [False] * len(self.graph)
        queue.append(start)
        visited[start] = True
        
        while queue:
            # Dequeue a vertex from the queue and print it
            vertex = queue.pop(0)
            print(vertex, end = " ")

            # Get all adjacent vertices of the dequeued vertex
            for neighbor in self.graph[vertex]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True

# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

print("Breadth-First Traversal starting from vertex 2:")
g.bfs(2)
