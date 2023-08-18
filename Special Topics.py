### ------ BFS algorithm in Python --------- ###

import collections

def bfs(graph, root):
		# initialize set that'll hold all visited nodes
		visited = set()
		# initialize queue that'll hold unvisited nodes
		# waiting to be visited
		queue = collections.deque([root])
		# we start by adding the root to the visited pile
		visited.add(root)

		# as long as the queue is non-empty, we run this
		while queue:
				# dequeue a vertex from queue
				# step 2 a bove (take the front item of the queue and add it to the visited list)
				vertex = queue.popleft()
				print(str(vertex) + " ", end = "")
				
				# if not visited, mark it as visited and
				# enqueue it
				for neighbour in graph[vertex]:
						if neighbour not in visited:
								visited.add(neighbour)
								queue.append(neighbour)

if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is Breadth First Traversal: ")
    bfs(graph, 0)