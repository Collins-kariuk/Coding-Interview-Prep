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
