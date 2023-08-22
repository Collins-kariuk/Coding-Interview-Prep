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
    

### ------ Tries in Python --------- ###
class Node:
	def __init__(self):
		self.key = None
		self.value = None
		self.children = {}
	
class Trie:
	def __init__(self):
		self.root = Node()
	
	def insert(self, word, value):
		currentWord = word
		currentNode = self.root

		while len(currentWord) > 0:
			if currentWord[0] in currentNode.children:
				currentNode = currentNode.children[currentWord[0]]
				currentWord = currentWord[1:]
			else:
				newNode = Node()
				newNode.key = currentWord[0]
				if len(currentWord) == 1:
					newNode.value = value
				currentNode.children[currentWord[0]] = newNode
				currentNode = newNode
				currentWord = currentWord[1:]
		
	def lookup(self, word):
		currentWord = word
		currentNode = self.root
		while len(currentWord) > 0:
			if currentWord[0] in currentNode.children:
				currentNode = currentNode.children[currentWord[0]]
				currentWord = currentWord[1:]
			else:
				return "Not in trie"
		
		if currentNode.value == None:
			return "None"
		return currentNode.value
	
	def printAllNodes(self):
		nodes = [self.root]
		while len(nodes) > 0:
			for letter in nodes[0].children:
				nodes.append(nodes[0].children[letter])
            # print(nodes.pop(0).key)

def makeTrie(words):
    trie = Trie()
    for word, value in words.items():
        trie.insert(word, value)
    return trie

		