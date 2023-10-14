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
            print(vertex, end=" ")

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

# print("Breadth-First Traversal starting from vertex 2:")
# g.bfs(2)


# ====================================================================================================
### DEPTH-FIRST SEARCH ###
"""
Depth-First Search (DFS) is an algorithm used to traverse or search through data structures like trees and graphs.
It explores as far as possible along each branch before backtracking.
DFS is used to find paths, connected components, and topological orderings in graphs.

Here's a step-by-step explanation of the DFS algorithm in Python:

1. Start from an initial node or vertex.
2. Mark the current node as visited.
3. Explore an adjacent unvisited node.
4. If an unvisited adjacent node exists, repeat steps 2 and 3 with that node as the current node.
5. If no unvisited adjacent node exists, backtrack to the previous node.
6. Repeat steps 2 to 5 until all nodes have been visited.
"""


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs(self, node, visited):
        # Mark the current node as visited and print it
        visited[node] = True
        print(node, end=" ")

        # Recur for all adjacent vertices
        for neighbor in self.graph[node]:
            if not visited[neighbor]:
                self.dfs(neighbor, visited)


# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

# print("Depth-First Traversal starting from vertex 2:")
# visited = [False] * len(g.graph)
# g.dfs(2, visited)
