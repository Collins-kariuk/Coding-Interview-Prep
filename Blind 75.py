### LINKED LISTS ###

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# --------- 1. Remove Nth Node from End of List - Leetcode 19 - Medium ------------
def removeNthFromEnd(head, n):
    # Create a dummy node and attach it to the head of the input list.
    dummy = ListNode(val = 0, next = head)
    
    # Initialize 2 pointers, first and second, to point to the dummy node.
    first = dummy
    second = dummy
    
    # Advances first pointer so that the gap between first and second is n nodes apart
    for i in range(n + 1):
        first = first.next
        
    # While the first pointer does not equal null move both first and second to maintain the gap and get nth node from the end
    while (first != None):
        first = first.next
        second = second.next
    
    # Delete the node being pointed to by second.
    second.next = second.next.next
    
    # Return dummy.next
    return dummy.next
    

### TREES ###

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# --------- 2. Invert Binary Tree - Leetcode 226 - Easy ------------
def invertTree(root):
    # check whether the root contains value and if not return none
    if root == None:
        return None

    # swap the left and right children
    # you have to store the left child in a variable first
    # to not lose it
    temp = root.left
    root.left = root.right
    root.right = temp

    # recursively call the method on the right and left children
    invertTree(root.left)
    invertTree(root.right)

    return root

# =================================================================== #

### GRAPHS ###

# --------- 3. Number of Islands - Leetcode 200 - Medium ------------
def numIslands(grid):
        # when the grid is empty
        if len(grid) == 0:
            return 0
        
        # initialize the number of islands
        islands = 0

        # set DS that'll store the visited islands
        visited = set()

        # number of rows and columns in the grid
        rows = len(grid)
        cols = len(grid[0])

        def bfs(r, c):
            """
            a breadth first search to check the number of islands
            by marking already visited islands so as to not
            forget which ones have already been visited
            """
            # bfs is an iterative algorithm that needs a DS,
            # which is normally a queue
            q = deque()
            # we add the island to the visited pile
            visited.add((r, c))
            # append the island we're at in the iteration in the
            # our bfs queue
            q.append((r, c))

            # traverse through the queue as long as it's non-empty
            # "expanding our island"
            while q:
                # the subgrid coord at the top of our queue
                row, col = q.popleft()
                # check the adjacent positions of the subgrid we're
                # looking at
                # generic LEFT, RIGHT, UP, and DOWN directions
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

                # 
                for dr, dc in directions:
                    # specific coordinates of neighbors
                    r = row + dr
                    c = col + dc
                    # check that the coordinates are in bounds
                    # check that it's land
                    # check that it's not visited yet
                    if r in range(rows) and c in range(cols) and grid[r][c] == '1' and (r, c) not in visited:
                        # add to queue because we also have to run bfs on this cell as well
                        q.append((r, c))
                        # mark it as visited so that we don't visit it twice
                        visited.add((r, c))

        # looping through each individual grid
        for r in range(rows):
            for c in range(cols):
                # if the subgrid is land and is not among the visited,
                # do a breadth-first-search on it and increment the
                # number of islands
                if grid[r][c] == "1" and (r, c) not in visited:
                    bfs(r, c)
                    islands += 1
        

        # the final number of islands
        return islands

# =================================================================== #

### ARRAYS & HASHING ###
# --------- 4. Contains Duplicate - Leetcode 217 - Easy ------------
def containsDuplicate(nums):
    # initialize a dictionary that'll store the running occurrences
    # of elements in nums
    dict_list = {}
    # loop through nums
    for num in nums:
        # if the number already exists in the dictionary, then nums
        # contains a duplicate since it means it was added in a
        # previous iteration
        if num in dict_list:
            return True
        # if the number does not already exist in the dictionary,
        # we add it to the dict, so that if we encounter it in a
        # following iteration, we'll know nums contains a duplicate
        else:
            dict_list[num] = 1
    # we've gone through all numbers in nums and have not found
    # nums to have a duplicate
    return False

# =================================================================== #

### HEAPS OR PRIORITY QUEUES ###
# --------- 5. Find Median from Data Stream - Leetcode 295 - Hard ------------
import heapq
class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        
        # two heaps, large, small, minheap, maxheap
        # heaps should be equal size
        self.small, self.large = [], []  # maxHeap, minHeap (python default)

    def addNum(self, num: int) -> None:
        # when the large heap is non-empty AND the number to be added is
        # greater than the largest number in the large heap, then we add
        # the number to the large heap
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        # otherwise, we just normally add it to the small heap
        # note that since Python's default implementation of a heap is a
        # minheap, we negate the numbers we need to add before adding them
        # to the small heap (the maxheap)
        else:
            heapq.heappush(self.small, -1 * num)

        # uneven sizes? we need the lengths of both heaps to be no more
        # than 1
        if len(self.small) > len(self.large) + 1:
            # the smaller heap is bigger than normal and so we need to
            # remove the smallest (the largest after multiplying by 1)
            # number and add it to the large heap to maintain length
            # parity
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)

    def findMedian(self) -> float:
        # when the length of the small heap is one greater than the
        # length of the bigger heap, we return the biggest value from
        # the small heap
        # note that the first index of a heap is always the root which
        # is the smallest value, but we return the negation of the
        # smallest value which will be the biggest value from the small
        # heap
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        # when the sizes of the small and large heaps are equal
        return (-1 * self.small[0] + self.large[0]) / 2



