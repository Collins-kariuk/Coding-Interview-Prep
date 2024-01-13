### LINKED LISTS ###
import heapq
from collections import defaultdict, deque
import collections


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# --------- 1. Remove Nth Node from End of List - Leetcode 19 - Medium ------------


def removeNthFromEnd(head, n):
    # Create a dummy node and attach it to the head of the input list.
    dummy = ListNode(val=0, next=head)

    # Initialize 2 pointers, first and second, to point to the dummy node.
    first = dummy
    second = dummy

    # Advances first pointer so that the gap between first and second is n nodes apart
    for i in range(n + 1):
        first = first.next

    # While the first pointer does not equal null move both first and second to maintain
    # the gap and get nth node from the end
    while first != None:
        first = first.next
        second = second.next

    # Delete the node being pointed to by second
    second.next = second.next.next

    # Return the head of the input list
    return dummy.next

# --------- 14. Reverse Linked List - Leetcode 209 - Easy ------------


def reverseList(head):
    # intialize the 2 needed pointers required to traverse through the linked list
    prev = None
    curr = head

    # continue with the loop as long as the current pointer does not point at a
    # null node
    while curr:
        # save the reference to the next node after the current one since
        # ,in the next iteration, it will serve as the current node
        placeholder = curr.next
        # changing the "direction of the arrow" or where the current node
        # points to
        curr.next = prev
        # advancing the previous pointer to be where the current pointer is
        prev = curr
        # advancing the current pointer to the placeholder we conveniently
        # saved earlier
        curr = placeholder

    # the new head of the reversed linked list will be the node the prev is pointing to
    return prev

# ------------- 17. Merge Two Sorted Lists - Leetcode 21 - Easy -----------------


def mergeTwoLists(l1, l2):
    # initialize a temp head that'll serve as a placeholder
    res = ListNode()
    # a pointer to the temp head
    tail = res

    # we continue with the loop as long as both pointers to the input linked lists
    # are non-null
    while l1 and l2:
        # when the value of the current node in l1 is smaller than that of the node in l2,
        # the node from l1 is appended to the resultant linked list.
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            # otherwise we add the node in l2 to the result linked list
            tail.next = l2
            l2 = l2.next
        # regardless, we advance the pointer to the result linked list
        tail = tail.next

    # if either of the pointers to the input linked lists are null, we add the
    # non-null linked list to the end of the result linked list
    if l1 is not None and l2 is None:
        tail.next = l1
    else:
        tail.next = l2
    # we return the next node of the temp head since the temp head is just a placeholder
    return res.next

# ----------------- 18. Linked List Cycle - Leetcode 141 - Easy --------------------


def hasCycle(head):
    """
    The essence of the method is that one pointer progresses more rapidly than the other.
    If there's a cycle in the linked list, the faster pointer will eventually catch up to the slower one.
    The moment they overlap is the indication of a cycle's presence.
    Both pointers are initially positioned at the head of the input linked list.
    """

    # using slow and fast pointers
    slow = head
    fast = head

    # we continue with the loop as long as the fast pointer is non-null and the
    # next pointer of the fast pointer is non-null
    # we include fast.next because if we don't, we'll get a null pointer exception
    # when we try to access fast.next.next
    while fast and fast.next:
        # slow pointer moves slower than the fast pointer
        slow = slow.next
        fast = fast.next.next
        # at a certain pointer in the iteration, the fast pointer will overlap the
        # slow pointer and when this happens we know there's a cycle
        if slow == fast:
            return True
    # once the loop exits, we know for sure that the linked list is linear since
    # either fast or fast.next would be null
    return False


# ----------------- 19. Reorder List - Leetcode 143 - Medium --------------------
def reorderList(head):
    """
    Modify the head directly, without returning any value.
    The core strategy involves dividing the input linked list into two sections.
    To facilitate alternating between the segments, pointers to the heads of both halves are necessary.
    The main hurdle is that the second half of the list must be reversed for straightforward reintegration,
    as backtracking is not possible in a singly linked list.
    """

    # find the middle point of the linked list
    slow = head
    fast = head.next

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # reverse second half
    # second half of the list
    second = slow.next
    # we split the linked list into 2 halves so instead of pointing the next pointer
    # of the middle node to the head of the second half of the linked list, we point
    # it to None thus effectively splitting the linked list into 2
    slow.next = None
    prev = None

    # reversing the second half (see reversing linked list question)
    while second:
        temp = second.next
        second.next = prev
        prev = second
        second = temp

    # merge the 2 halves
    first = head
    # Once the execution of the second while loop concludes, the prev pointer will be
    # at the head of the now-reversed second half of the linked list
    # This occurs because the second pointer, which traverses the second half of the
    # list, reaches Null
    second = prev
    # The merging process goes on until either the first or second pointer becomes null
    # However, given that the second half of the list might be shorter, we can
    # base our continuation condition primarily on the status of the second pointer
    while second:
        # we store the references of the next nodes in separate variables since we know
        # we are going to break the links as we traverse through both halves
        temp1 = first.next
        temp2 = second.next

        first.next = second
        second.next = temp1

        # advance our pointers forward in the respective halves
        # this is easy since we saved the references to the old/prior nexts
        first = temp1
        second = temp2


# ------------- 42. Merge k Sorted Lists - Leetcode 23 - Hard -----------------
def mergeKLists(lists):
    """
    the gist of the solution is that we merge the lists 2 at a time
    we do this until we have one merged list
    divide and conquer
    """

    # the edge cases
    # if the input list is empty, we return None
    if not lists or len(lists) == 0:
        return None

    # while the length of the input list is greater than 1, we merge the lists 2 at a time
    while len(lists) > 1:
        # the list that'll store the merged lists
        mergedLists = []

        # we merge the lists 2 at a time
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            # the length of the input list could be odd and so we need to check whether
            # the index we're trying to access is in bounds
            if i + 1 < len(lists):
                l2 = lists[i + 1]
            else:
                # if it's not in bounds, we just set the second list to be None which is
                # still fine since we can merge a list with None
                l2 = None
            # we append the merged lists to the list that'll store the merged lists
            mergedLists.append(mergeTwoLists(l1, l2))
        # we update the input list to be the list that stores the merged lists
        lists = mergedLists
    # we return the first element of the input list since it'll be the merged list
    return lists[0]


# =================================================================== #

### TREES ###

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# --------- 2. Invert Binary Tree - Leetcode 226 - Easy ------------


def invertTree(root):
    # base case
    # check whether the root contains value and if not return none
    if root == None:
        return None

    # Exchange the left and right children of the node
    # To ensure the left child is not lost in the process, it's crucial
    # to first save it in a temporary variable
    temp = root.left
    root.left = root.right
    root.right = temp

    # recursive case
    # recursively on the right and left children
    invertTree(root.left)
    invertTree(root.right)

    return root

# ---------- 20. Maximum Depth of Binary Tree - Leetcode 104 - Easy -------------


def maxDepth(root):
    # base case
    # when we reach a null node, we return 0 since the depth of a null node is 0
    if root is None:
        return 0

    # recursive case
    # calculate the depth of the left part of the binary tree
    leftDepth = maxDepth(root.left)
    # calculate the depth of the right part of the binary tree
    rightDepth = maxDepth(root.right)
    # the maximum depth of the binary tree will be the larger of the left and right
    # depths plus 1 since we're counting the current node as well, i.e., the root
    return max(leftDepth, rightDepth) + 1

# --------------- 26. Same Tree - Leetcode 100 - Easy --------------


def isSameTree(p, q):
    # base case part 1
    # when both nodes are null, we can consider them to be the
    # same tree and hence we return true
    if p == None and q == None:
        return True

    # base case part 2
    # If one of the nodes is null while the other isn't, or if the values
    # of the nodes don't match, we recognize these as different trees
    # In either of these cases, we return false, indicating a mismatch
    if (p == None or q == None) or (p.val != q.val):
        return False

    # the recursive case
    # For the trees to be considered identical, both the left and right sides must
    # be exactly the same. This means if the left subtrees of p and q match but
    # their right subtrees do not, the trees are deemed different
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

# --------------- 27. Subtree of Another Tree - Leetcode 572 - Easy --------------


def isSubtree(root, subroot):
    # check 26. Same Tree - Leetcode 100 - Easy
    def isSameTree(p, q):
        if p == None and q == None:
            return True
        if (p == None or q == None) or (p.val != q.val):
            return False
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

    # When the subroot is null, it's technically a subtree of any other tree, as a null
    # node can always be found among the leaf children in any tree
    if subroot is None:
        return True

    # In the case where the root is null, it cannot contain any subtree other than itself
    # (which is also null). This scenario has already been accounted for previously
    if root is None:
        return False

    # if the root and subroot are the same, then we return True
    if isSameTree(root, subroot):
        return True

    # If the above conditions are not met, we proceed to recursively examine the left
    # and right subtrees of the root to determine if they include the subroot
    # This approach is valid because we've established that the subroot is distinct from the root.
    # Therefore, we need to investigate whether the subroot matches either the left or right subtree of the root.
    return isSubtree(root.left, subroot) or isSubtree(root.right, subroot)

# --------------- 28. Lowest Common Ancestor of a Binary Search Tree - Leetcode 235 - Medium --------------


def lowestCommonAncestor(root, p, q):
    while True:
        if root.val < p.val and root.val < q.val:
            root = root.right
        elif root.val > p.val and root.val > q.val:
            root = root.left
        else:
            return root

# --------------- 29. Binary Tree Level Order Traversal - Leetcode 102 - Medium --------------


def levelOrder(root):
    # initialize the result
    res = []
    # initialize queue for Breadth First Search (BFS)
    q = collections.deque()
    # add the root to the queue
    q.append(root)

    # run BFS while queue is nonempty
    while q:
        # get the length of the queue
        # get number of nodes that are in the queue at a given point/currently
        # ensures that we go through the queue one level at a time
        qLen = len(q)
        level = []
        # loop through every value currently in the queue
        for i in range(qLen):
            # pop nodes from the left of the queue (FIFO)
            node = q.popleft()
            # it's technically possible that the node could be null, so we have to check that
            # the popped node is non-null before proceeding
            if node:
                # append the node's value to the current level
                level.append(node.val)
                # add the children of the popped node
                q.append(node.left)
                q.append(node.right)
        # if the level is non-empty, we append it to the result
        if level:
            res.append(level)
    # return the result
    return res


# --------------- 43. Validate Binary Search Tree - Leetcode 98 - Medium --------------
def isValidBST(root):
    # the gist of the solution is that we need to check whether the current node's value
    # is between the minimum and maximum values
    # we do this by recursively calling the function on the left and right subtrees
    # and updating the minimum and maximum values as we go along
    def helper(node, minVal, maxVal):
        # the base case
        # if the node is null, we return True
        if not node:
            return True
        # if the node's value is less than the minimum value or greater than the maximum
        # value, we return False
        if node.val <= minVal or node.val >= maxVal:
            return False
        # the recursive case
        # we call the function on the left and right subtrees
        # we update the minimum and maximum values as we go along
        return helper(node.left, minVal, node.val) and helper(node.right, node.val, maxVal)

    # we start the recursive function call with the root node
    # we set the minimum and maximum values to be the minimum and maximum values of a
    # 32-bit signed integer
    return helper(root, -2**31, 2**31 - 1)


# --------------- 44. Kth Smallest Element - Leetcode 230 - Medium --------------
def kthSmallest(root, k):
    # the gist of the solution is that we need to do an inorder traversal of the binary
    # search tree and return the kth element
    # we do this iteratively using a stack

    # initialize a counter to keep track of the number of nodes we've visited so far
    # which we'll see if the number of nodes we're going to pop off the stack
    n = 0
    # initialize a stack to keep track of the nodes we've visited so far
    stack = []
    # initialize a pointer to the root node
    cur = root

    # we continue with the loop as long as the current node is non-null or the stack
    # is non-empty
    while cur is not None or len(stack) != 0:
        # we traverse to the leftmost node
        while cur is not None:
            # as we traverse to the leftmost node, we add the nodes we visit to the stack
            stack.append(cur)
            # we keep traversing to the leftmost node
            cur = cur.left

        # we pop the topmost node off the stack when we hit a left null node
        # assigning the top element of the stack to cur allows us to travel back to the
        # parent node of the left null node and carry on with the inorder traversal
        # by doing the same thing we did with the left subtree with the right subtree
        cur = stack.pop()
        # we increment the counter since we've hit a null node
        n += 1

        # we check whether we've hit the kth node
        # this allows us to return the value of the kth node as soon as we hit it
        if n == k:
            return cur.val

        # we traverse to the right subtree
        cur = cur.right


# --------------- 45. Construct Binary Tree from Preorder and Inorder Traversal - Leetcode 105 - Medium --------------
def buildTree(preorder, inorder):
    # the gist of the solution is that we need to build the tree recursively
    # we do this by using the preorder traversal to find the root node and
    # the inorder traversal to find the left and right subtrees of the root node
    # we do this recursively until we've built the entire tree

    # the base case is when either the preorder or inorder traversal is empty
    # in which case we return None
    if len(preorder) == 0 or len(inorder) == 0:
        return None

    # the root node is the first element in the preorder traversal (a given)
    root = TreeNode(preorder[0])
    # find the index of the root node in the inorder traversal list
    mid = inorder.index(preorder[0])

    # recursively build the left and right subtrees of the root node
    # Notice that everything from the beginning of the inorder
    # traversal we're at till the middle index form the nodes of the left subtree
    # (though we don't include the middle index) and everything from the middle to
    # the end of the inorder list form the nodes of the right subtree and so we can
    # use this to our advantage because at any given time, we know the number of nodes
    # in the left and right subtrees and so we can use this to find the left and right
    # subtrees in the preorder traversal list (this handles the understanding of the
    # recursive calls for inorder)

    # for the preorder traversal, we know that the first element is the root node and
    # so we can use this to our advantage to find the left and right subtrees in the
    # preorder traversal list since the next element after the root node is the root
    # node of the left subtree and since we know however many number of nodes that are
    # in the left subtree (0:mid+1), we can use that to find the root node of the right
    # subtree (mid+1::) (this handles the understanding of the recursive calls for preorder)

    root.left = buildTree(preorder[1:mid + 1], inorder[0:mid])
    root.right = buildTree(preorder[mid + 1::], inorder[mid + 1::])

    return root


# --------------- 46. Binary Tree Maximum Path Sum - Leetcode 124 - Hard --------------
def maxPathSum(root):
    # the gist of the solution is that we need to find the maximum path sum we do this
    # by recursively calling the function on the left and right subtrees and updating
    # the maximum path sum as we go along we also need to check whether the maximum
    # path sum includes the root node or not

    # initially,we set the maximum path sum to be the value of the root node we're
    # looking at
    res = [root.val]

    # return max path sum without splitting
    def dfs(root):
        # base case: when we hit a null node, we return 0
        if root is None:
            return 0

        # recursive case: run the DFS on the left and right subtrees
        leftMax = dfs(root.left)
        rightMax = dfs(root.right)

        # the maximum sum we can get without splitting from the input left and right
        # nodes, respectively we need to check whether the maximum sum we can get
        # without splitting from the input left and right nodes, respectively, is
        # negative or not because if it is, we don't want to include it in our maximum
        # path sum
        leftMax = max(leftMax, 0)
        rightMax = max(rightMax, 0)

        # compute the max path sum WITH splitting from the input node
        res[0] = max(res[0], root.val + leftMax + rightMax)

        # return the max path sum WITHOUT splitting from the input node
        return root.val + max(leftMax, rightMax)

    # run the DFS on the root node
    dfs(root)
    # return the maximum path sum
    return res[0]

# =================================================================== #

### GRAPHS ###

# --------- 3. Number of Islands - Leetcode 200 - Medium ------------


def numIslands(grid):
    # when the grid is empty
    if len(grid) == 0:
        return 0

    # initialize the number of islands
    islands = 0

    # set that'll store the visited islands
    visited = set()

    # number of rows and columns in the grid
    rows = len(grid)
    cols = len(grid[0])

    def bfs(r, c):
        """
        Conducting a breadth-first search to count the number of islands, while keeping track of the
        islands already visited. This ensures that we don't mistakenly revisit the same islands.
        """

        # bfs is an iterative algorithm that needs a data structure, which is normally a queue
        q = deque()
        # we add the island to the visited pile
        visited.add((r, c))
        # append the island we're at in the iteration in our bfs queue
        q.append((r, c))

        # traverse through the queue as long as it's non-empty thus "expanding our island"
        while q:
            # the subgrid coord at the top of our queue
            row, col = q.popleft()
            # check the adjacent positions of the subgrid we're looking at generic LEFT, RIGHT, UP,
            # and DOWN directions
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

            for dr, dc in directions:
                # specific coordinates of neighbors
                r = row + dr
                c = col + dc
                # check that the coordinates are in bounds
                # check that it's land
                # check that it's not visited yet
                if (r in range(rows) and c in range(cols)) and grid[r][c] == '1' and (r, c) not in visited:
                    # add to queue because we also have to run bfs on this cell as well
                    q.append((r, c))
                    # mark it as visited so that we don't visit it twice
                    visited.add((r, c))

    # looping through each individual grid
    for r in range(rows):
        for c in range(cols):
            # if the subgrid is land and is not among the visited, do a breadth-first-search on it and
            # increment the number of islands
            if grid[r][c] == "1" and (r, c) not in visited:
                bfs(r, c)
                islands += 1

    # the final number of islands
    return islands


# --------- 47. Clone Graph - Leetcode 133 - Medium ------------
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def cloneGraph(node):
    # the gist of the solution is that we need to clone the graph recursively
    # we do this by using a dictionary to keep track of the nodes we've visited
    # and their corresponding clones
    # we do this recursively until we've cloned the entire graph
    # the key:value pairs in this dictionary is oldNode:cloneNode
    oldToNew = {}

    # the recursive function which is a depth-first-search
    def dfs(node):
        # the base case
        # if the node we're looking at is already in the dictionary, we return
        # its clone, which is its value in the dictionary
        if node in oldToNew:
            return oldToNew[node]

        # the recursive case
        # we create a clone of the node we're looking at
        # note that the creation of the copy of the node is incomplete because in addition to assigning a value to the node, we also need to assign its neighbors
        copy = Node(node.val)
        # we add the node we're looking at to the dictionary mapping old nodes to new nodes
        oldToNew[node] = copy

        # we loop through the neighbors of the node we're looking at
        for neighbor in node.neighbors:
            # we add the neighbors of the node we're looking at to the neighbors of the copy of the node we're looking at
            copy.neighbors.append(dfs(neighbor))
        # the depth-first-search returns the copy of the node we're looking at if it's not in the dictionary
        return copy

    # we start the recursive function call with the input node
    if node is not None:
        return dfs(node)
    # edge case is when the node is null
    else:
        return None


# --------- 48. Pacific Atlantic Water Flow - Leetcode 417 - Medium ------------
def pacificAtlantic(heights):
    # the gist of the solution is that we need to find the cells that can flow to both the Pacific and Atlantic oceans
    # we do this by using 2 sets to keep track of the cells that can flow to the Pacific and Atlantic oceans, respectively
    # we do this by using a depth-first-search to find the cells that can flow to the Pacific and Atlantic oceans, respectively
    # we do this recursively until we've found all the cells that can flow to the Pacific and Atlantic oceans, respectively

    # store the number of rows and columns in the input matrix
    ROWS = len(heights)
    COLS = len(heights[0])

    # initialize the sets that'll store the cells that can flow to the Pacific and Atlantic oceans, respectively
    pacific = set()
    atlantic = set()

    # the recursive function which is a depth-first-search that finds the cells that can flow to the Pacific and Atlantic
    # oceans, respectively
    def dfs(r, c, visit, prevHeight):
        """
        r: row
        c: column
        visit: set that'll store the cells that can flow to the Pacific and Atlantic oceans, respectively (depending on how
        the dfs function is called outside of itself, i.e., visit is a generic name for pacific and atlantic)
        prevHeight: the height of the previous cell
        """
        # the base case:
        # - When we've already visited the cell
        # - When the cell is out of bounds
        # - When the height of the cell is less than the height of the previous cell. For this, notice that we're starting
        # from the oceans and proceeding inwards inland, so our scenario changes from wishing that the neighboring cell have
        # a smaller height value to wanting the neighboring cell to have a larger height value
        if ((r, c) in visit or r < 0 or c < 0 or r == ROWS or c == COLS or heights[r][c] < prevHeight):
            return

        # add the cell to the set that'll store the cells that can flow to the Pacific and Atlantic oceans, respectively
        visit.add((r, c))
        # call dfs on the neighboring cells (WEST, EAST, SOUTH, NORTH)
        dfs(r + 1, c, visit, heights[r][c])
        dfs(r - 1, c, visit, heights[r][c])
        dfs(r, c + 1, visit, heights[r][c])
        dfs(r, c - 1, visit, heights[r][c])

    # we start the recursive function call with the input node
    # notice that the first and last rows will border the Pacific and Atlantic oceans, respectively and so we can run a DFS
    # from each cell in these rows to figure out which cells (inwardly) can flow to the Pacific and Atlantic oceans, respectively
    for c in range(COLS):
        # call the dfs function on the first and last rows
        # since water can flow between cells of equal heights, the value for prevHeight is set to the height of the first
        # and last cells in the first and last rows, respectively
        dfs(0, c, pacific, heights[0][c])
        dfs(ROWS - 1, c, atlantic, heights[ROWS - 1][c])

    # also notice that the first column and last column will border the Pacific and Atlantic oceans, respectively and so we
    # can run a DFS from each cell in these columns to figure out which cells (inwardly) can flow to the Pacific and Atlantic
    # oceans, respectively
    for r in range(ROWS):
        dfs(r, 0, pacific, heights[r][0])
        dfs(r, COLS - 1, atlantic, heights[r][COLS - 1])

    # initialize the result
    res = []

    # loop through the cells in the input matrix
    for r in range(ROWS):
        for c in range(COLS):
            # figure out which cells can flow to both the Pacific and Atlantic oceans, i.e., which (r, c) pairs are in both
            # the pacific and atlantic sets and add them to the result
            if (r, c) in pacific and (r, c) in atlantic:
                res.append([r, c])
    return res

# --------- 48. Course Schedule - Leetcode 207 - Medium ------------


def canFinish(numCourses, prerequisites):
    # the gist of the solution is that we need to check whether the courses can be completed
    # we do this by using a dictionary to keep track of the prerequisites of each course
    # we do this by using a set to keep track of the courses we've visited
    # we do this recursively until we've checked whether the courses can be completed

    # initialize a dictionary that'll store the prerequisites of each course
    prereqMap = {i: [] for i in range(numCourses)}

    # loop through the prerequisites and add them to the dictionary to the corresponding course
    # that it's a prerequisite for
    for crs, prereq in prerequisites:
        prereqMap[crs].append(prereq)

    # initialize a set that'll store the courses we've visited
    visitSet = set()

    # the recursive function which is a depth-first-search that checks whether the courses can be completed
    def dfs(crs):
        # if the course is in the set of courses we've visited, it means that we've encountered a cycle
        # and so we return False
        if crs in visitSet:
            return False
        # if the course has no prerequisites, we return True since it means that we've reached the end
        # and we can take the course
        if prereqMap[crs] == []:
            return True

        # add the course to the set of courses we've visited
        # visit set contains all courses allong the current DFS path
        visitSet.add(crs)
        # run dfs on the prerequisites of the course
        for prereq in prereqMap[crs]:
            # if the prerequisites of the course cannot be completed, we return False immediately
            # we don't need to check the other prerequisites of the course
            if not dfs(prereq):
                return False
        # remove the course from the set of courses we've visited
        # we've already checked the prerequisites of the course and so we can remove it from the set
        # i.e., we've finished visiting the course
        visitSet.remove(crs)
        # we set the prerequisites of the course to be an empty list since we've already checked them
        # and so we don't need to check them again
        prereqMap[crs] = []
        return True

    # loop through the courses and run dfs on them
    # the reason we need to loop through the courses is because there could be multiple courses
    # that are not connected, i.e., separate disconnected directed graphs
    for crs in range(numCourses):
        # if one of the courses cannot be completed, we return False immediately
        if not dfs(crs):
            return False
    return True


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

# using a set


def containsDuplicate(nums):
    hashset = set()

    for n in nums:
        if n in hashset:
            return True
        hashset.add(n)
    return False

# ---------------- 9. Two Sum - Leetcode 1 - Easy -------------------


def twoSum(nums, target):
    # dictionary that'll store already visited numbers in nums alongside
    # their indices as key:value pairs
    # conveniently, the values will be the index of each number
    twoSumDict = {}

    # looping through nums, where each number in each iteration will act as num1
    for i in range(len(nums)):
        # the second number, num2, that when added to num1 will produce target
        num2 = target - nums[i]
        # when num2 is already in the dictionary, it means we've already found
        # our 2 numbers and we can return their indices as a list
        if num2 in twoSumDict:
            return [i, twoSumDict[num2]]
        # otherwise we add the number, at the current iteration, into the
        # dictionary which will later serve as num2
        else:
            twoSumDict[nums[i]] = i

# ---------------- 11. Top K Frequent Elements - Leetcode 347 - Medium -------------------


def topKFrequent(nums, k):
    # function to count the frequency of numbers in a list of numbers
    def dictify(nums):
        numsDict = {}
        for num in nums:
            if num in numsDict:
                numsDict[num] += 1
            else:
                numsDict[num] = 1
        return numsDict

    # dictionary containing key:value pairs of the frequency of the occurrences of numbers
    # in nums
    count = dictify(nums)

    # another way to initialize the dictionary
    # for n in nums:
    #         count[n] = 1 + count.get(n, 0)

    # our "bucket"
    # essentially, frequencies acts as a measure of the question, "how many elements occur
    # i number of times?" where i is dictated by the length of nums
    # essentially, this provides an easy control of the length of the list that controls
    # the frequencies of the numbers in nums
    # if you may imagine another scenario where the length of frequencies is controlled by
    # the actual elements in nums instead of their frequencies, we would have the length of
    # the frequencies list being controlled by the largest number in nums which would not
    # be a good use of memory since the largest number in nums could be quite a large number,
    # say 1,000,000
    frequencies = [[] for i in range(len(nums) + 1)]
    # frequencies = [[]] * (len(nums) + 1)

    # loop through the numbers and their respective counts in the dictionary
    for num, count in count.items():
        frequencies[count].append(num)

    # initialize our results array
    res = []
    # loop through the nested list that is frequencies
    # notice that we loop starting from the end since we want the top k frequent elements
    for i in range(len(frequencies) - 1, 0, -1):
        # looping through each individual sublist
        for num in frequencies[i]:
            # add the top number in res
            res.append(num)
            # we have to stop eventually, and this will be signified when the length of res
            # is equal to k
            # we need not go further when it is so
            if len(res) == k:
                return res

# --------- 25. Valid Anagram - Leetcode 242 - Easy ------------


def isAnagram(s, t):
    def dictify(s):
        wordCounter = {}
        for char in s:
            if char in wordCounter:
                wordCounter[char] += 1
            else:
                wordCounter[char] = 1
        return wordCounter

    sDict = dictify(s)
    tDict = dictify(t)
    return sDict == tDict


def isAnagram(s, t):
    # if the lengths of the strings are not equal, they can't be anagrams
    if len(s) != len(t):
        return False

    # initialize 2 dictionaries that'll store the number of occurrences of each
    # character in the strings
    countS, countT = {}, {}

    # loop through the strings and count the number of occurrences of each character
    for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0)
        countT[t[i]] = 1 + countT.get(t[i], 0)
    return countS == countT

# --------- 28. Group Anagrams - Leetcode 49 - Medium ------------


def groupAnagrams(strs):
    res = defaultdict(list)

    # visit each string in strs
    for s in strs:
        # initialize a specific count list for that specific list
        # since we have not started counting the number of occurrences
        # of the characters in each string, we feel the count with zeros
        # 26 for every letter in the alphabet
        # count[0] -> 'a', count[1] -> 'b', count[2] -> 'c'...
        # count[25] -> 'z'
        count = [0] * 26

        # visit each character in the string
        for char in s:
            # count the occurrence of each letter/character in the string
            # using ord helps us assign each letter in the alphabet to
            # its corresponding index in count
            count[ord(char) - ord('a')] += 1
        # append the string that corresponds to that particular count list
        # to the value that corresponds to the specific count key in the
        # default dictionary
        res[tuple(count)].append(s)

    # we just want all the values in list forms
    return list(res.values())


# --------- 37. Product of Array Except Self - Leetcode 238 - Medium ------------
def productExceptSelf(nums):
    """
    the gist of the solution is that we need to calculate the product of all
    the numbers to the left of a number and the product of all the numbers
    to the right of a number
    we can do this by using 2 arrays, one that stores the product of all the
    numbers to the left of a number and another that stores the product of
    all the numbers to the right of a number
    we then multiply the corresponding elements in the 2 arrays to get the
    product of all the numbers except the number at the current index
    """

    # initialize the prefix and postfix arrays
    pre = [1]
    post = [1]

    # calculate the prefix array
    preMult = 1
    for i in range(len(nums)):
        preMult *= nums[i]
        pre.append(preMult)
    # we remove the last element since it's not needed in the calculation
    pre.pop()

    # calculate the postfix array
    postMult = 1
    # we loop backwards
    for i in range(len(nums) - 1, 0, -1):
        postMult *= nums[i]
        post.append(postMult)

    # reverse the postfix array
    post = post[::-1]

    # first of all, initialize the result array to be all zeros because we
    # need to multiply the corresponding elements in the prefix and postfix
    # arrays and to index into the result array, we need to have something in
    # the result array so the zeros are the placeholders
    res = [0] * len(pre)
    for i in range(len(pre)):
        res[i] = pre[i] * post[i]

    return res


# --------- 38. Longest Consecutive Sequence - Leetcode 128 - Medium ------------
def longestConsecutive(nums):
    # convert the list to a set to eliminate duplicates
    numSet = set(nums)
    # variable that'll store the longest consecutive sequence
    longest = 0

    # loop through the numbers
    for n in nums:
        # check whether the number before the current number is the start of a new sequence
        # if it is, then we can calculate the length of that particular sequence
        # we do this by checking whether the number before the current number is in the set
        # if it is, then we know that the current number is not the start of a new sequence
        # and we can safely skip it
        if n - 1 not in numSet:
            # variable that'll store the length of the current sequence
            length = 0
            # we calculate the length of the current sequence by checking whether the number
            # after the current number is in the set and if it is, we increment the length
            # notice that since we initialize length to 0, we ensure that we count the
            # current start of the sequence as well
            while n + length in numSet:
                length += 1
            # we update the longest sequence if the current sequence is longer
            longest = max(length, longest)

    return longest


# ---------- 53. Encode and Decode Strings - Leetcode 659 - Medium -------------


def encode(strs):
    """
    Encodes a list of strings to a single string.
    """
    res = ''
    for string in strs:
        res += str(len(string)) + '#' + string
    return res


def decode(strs):
    """
    Decodes a single string to a list of strings.
    """
    res = []
    i = 0

    # iterate through the string
    while i < len(strs):
        # another iterator/pointer which is used to find the length of the string, i.e.,
        # useful in finding the leading digits that signify the length of the string
        j = i
        # could have that the string has a length greater than 9
        while strs[j] != '#':
            j += 1
        # length is the length of the string we are going to append to the result
        length = int(strs[i:j])
        # append the string to the result
        # we don't include the j (#) as part of the string which is why we start at j + 1
        res.append(strs[j + 1:j + length + 1])
        # the start of the next string
        # after this assignment, i will point to the first digit of the length of the next string
        i = j + length + 1
    return res


# =================================================================== #


### HEAPS OR PRIORITY QUEUES ###
# --------- 5. Find Median from Data Stream - Leetcode 295 - Hard ------------


class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        # two heaps, large, small, minheap, maxheap
        # heaps should be equal size
        self.small = []  # maxheap
        self.large = []  # minHeap (Python's default)

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

        # uneven sizes?
        # we need the lengths of both heaps to be no more than 1
        if len(self.small) > len(self.large) + 1:
            # the smaller heap is bigger than normal and so we need to
            # remove the smallest (the largest after multiplying by 1)
            # number and add it to the large heap to maintain length parity
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

# ================================================================================= #

### SLIDING WINDOW ###
# --------- 6. Best Time to Buy and Sell Stock - Leetcode 121 - Easy ------------


def maxProfit(prices):
    # initialize pointers
    l = 0
    r = 1
    # initialize variable that's gonna store the maximum profit
    maxProfit = 0

    # continue as long as the right pointer doesn't go out of bounds of the length
    # of the prices list
    while r < len(prices):
        # it's futile to calculate the profit if the value at the right pointer is
        # less than the value at the left pointer, so it means we've encountered a
        # value which is less than the one at the left pointer and would therefore
        # be better suited to calculate the maximum profit
        # we therefore change the left pointer to where the right pointer is and
        # increment the right pointer by one
        if prices[l] >= prices[r]:
            # the reason why we change the left pointer to where the right pointer
            # is because we need to keep track of the minimum value in the prices list
            # so that we can calculate the maximum profit
            l = r
            r += 1
        else:
            # otherwise we calculate the current profit normally
            currProfit = prices[r] - prices[l]
            # the maximum profit will be the larger profit between the previous max
            # profit and the just calculated current profit
            maxProfit = max(maxProfit, currProfit)
            # increment the right pointer by one
            r += 1
    return maxProfit

# --------- 7. Longest Substring Without Repeating Characters - Leetcode 3 - Medium ------------


def lengthOfLongestSubstring(s):
    # create a set that'll store all unique non-repeating characters
    charSet = set()
    # initialize left pointer
    l = 0
    # result that'll hold the longest substring without repeating characters
    res = 0

    for r in range(len(s)):
        # if the character at the right pointer is already in the set, it means
        # that we've encountered a repeating character and we need to remove the
        # character at the left pointer from the set and advance the left pointer
        # rightwards
        while s[r] in charSet:
            # 1. remove the character at the left pointer from the set
            # this is done because we need to keep track of the characters in the
            # substring we're currently looking at
            charSet.remove(s[l])
            # 2. advance the left pointer rightwards
            # this is done because we need to keep track of the characters in the
            # substring we're currently looking at
            # we also need to advance the left pointer rightwards because we need
            # to remove the character at the left pointer from the set
            l += 1
        # add the character at the right pointer to the set
        # this is done regardless of whether the character is repeated or not
        # because we need to keep track of the characters in the substring
        # we're currently looking at
        charSet.add(s[r])
        # the maximum length of the substring without repeating characters will be the
        # larger of the previous length and the current length
        # the current length is just the difference between the right and left pointers
        # plus one since we're dealing with 0-indexing
        res = max(res, r - l + 1)
    return res


# --------- 40. Longest Repeating Character Replacement - Leetcode 424 - Medium ------------
def characterReplacement(s, k):
    # initialize a dictionary that'll store the running occurrences
    count = {}
    res = 0
    # initialize left pointer
    l = 0

    # loop through the input string with the iterator acting as the right pointer
    for r in range(len(s)):
        # increment the count of the character at the right pointer in your dictionary
        count[s[r]] = count.get(s[r], 0) + 1
        # check whether a character replacement can even be made
        # the logic of this is that if the length of the current substring minus the
        # maximum count of a character in the current substring is greater than k, then
        # we know that we can't make a character replacement
        # it makes sense to replace the characters that are NOT the most frequent character
        # in the current substring because we can replace them with the most frequent character
        # replacements (k) in the current substring and still have a valid substring
        while r - l + 1 - max(count.values()) > k:
            # when this occurs, we need to decrement the count of the character at the left
            # pointer in the dictionary since we're going to move the left pointer rightwards
            # in order to make a valid substring that can be made with k character replacements
            # (is less than k)
            count[s[l]] -= 1
            l += 1
        # the maximum length of the substring with repeating characters will be the larger of
        # the previous window length and the current sliding window length
        res = max(res, r - l + 1)
    return res


# --------- 41. Minimum Window Substring - Leetcode 76 - Hard ------------
def minWindow(s, t):
    # edge case when the input string is empty
    if t == "":
        return ""

    # initialize a dictionary that'll store the running occurrences
    # of characters in t
    countT = {}
    # initialize a dictionary that'll store the running occurrences
    # of characters in the current window
    window = {}

    # count the occurrences of characters in t
    for c in t:
        countT[c] = 1 + countT.get(c, 0)

    # initialize variables that'll store the number of characters that we
    # have in the current window and the number of characters that we need
    # in the current window
    have = 0
    # the number of characters that we need in the current window is just the
    # length of the dictionary that stores the running occurrences of
    # characters in t
    need = len(countT)

    # initialize variables that'll store the result
    res = 0
    res = [-1, -1]
    # initially, the length of the result is infinity since we haven't found
    # a valid substring yet
    resLen = float("infinity")
    # initialize left pointer
    l = 0

    # loop through the input string with the iterator acting as the right pointer
    for r in range(len(s)):
        # save the character at the right pointer in a variable
        c = s[r]
        # increment the count of the character at the right pointer (saved in the variable above) in your dictionary
        window[c] = 1 + window.get(c, 0)

        # check whether the character at the right pointer is in the dictionary that stores the running occurrences
        # of characters in t AND whether the number of occurrences of the character at the right pointer in the
        # current window is equal to the number of occurrences of the character at the right pointer in t
        # if both conditions are satisfied, it means that we have a character that we need in the current window
        # and we can increment the number of characters that we have in the current window
        # note that c could be in the dictionary that stores the running occurrences of characters in t but
        # the number of occurrences of c in the current window could be less than the number of occurrences
        # of c in t, so we need to check that as well
        if c in countT and window[c] == countT[c]:
            have += 1

        # while we have all the characters that we need in the current window
        while have == need:
            # update our potential result
            if (r - l + 1) < resLen:
                # the potential result is encapsulated within the bounds of the indices of these pointers in res
                res = [l, r]
                # update the length of the potential result
                resLen = r - l + 1
            # pop from the left of our window and update the number of characters that we have in the current window
            # notice that by popping from the left of our window, there is potential to lose a character that we need
            # in the current window, so we need to check that as well
            # we perform this check by checking whether the count of the character at the left pointer in the window
            # we've popped from is less than the count of the character at the left pointer in t
            # recall that for have to be equal to need, the number of occurrences of the character each character in t
            # in the current window must be equal to the number of occurrences of the character in t
            window[s[l]] -= 1
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1
            # advance the left pointer rightwards to look at other potential substrings
            l += 1
    # the result is the substring that is encapsulated within the bounds of the indices of these pointers in res
    l, r = res
    # if the length of the result is not infinity, it means that we've found a valid substring
    # otherwise, we return an empty string
    if resLen != float("infinity"):
        return s[l:r + 1]
    else:
        return ""


# =================================================================== #

### BINARY SEARCH ###
# --------- 8. Find Minimum in Rotated Sorted Array - Leetcode 153 - Medium ------------


def findMin(nums):
    # variable that'll store the current minimum
    res = nums[0]
    # intiaialize pointers
    l = 0
    r = len(nums) - 1

    while l <= r:
        # if the number at the left pointer is less than the one at the right pointer,
        # it means that nums is already sorted and we can safely return the number at
        # the left pointer or the current minimum, whichever is smaller
        if nums[l] < nums[r]:
            res = min(res, nums[l])
            break

        # calculation of the location of the middle pointer
        mid = (l + r) // 2
        # before further comparison, the number at the middle pointer will serve as
        # the minimum
        res = min(res, nums[mid])
        # If the central element is greater than or equal to the element at the left pointer,
        # it indicates that the left segment of the sublist is already sorted. Due to the
        # array's rotation, searching in the left segment is not logical, as it will always
        # contain larger values compared to the right segment. Therefore, our search should
        # concentrate on the right segment of the array.
        if nums[mid] >= nums[l]:
            l = mid + 1
        # the opposite holds
        else:
            r = mid - 1
    return res

# ---------- 10. Search in Rotated Sorted Array - Leetcode 33 - Medium -------------


def search(nums, target):
    # initialize your pointers
    l = 0
    r = len(nums) - 1

    # as long as the pointers don't cross each other, continue with the
    while l <= r:
        # calculated the middle pointer
        mid = (l + r) // 2
        # direct return if the target is equal to the number at the middle pointer
        if target == nums[mid]:
            return mid

        # left sorted portion
        # if the number at the middle is greater than the number at the left pointer,
        # we are at the left sorted portion
        if nums[l] <= nums[mid]:
            # if the target is greater than the number at the middle OR if the target is
            # less than the number at the left pointer, there is no point in looking
            # at the left sorted portion, so we update our pointers to concentrate our
            # search on the right sorted portion
            if target > nums[mid] or target < nums[l]:
                l = mid + 1
            # otherwise, our target is surely in the left sorted portion and we change
            # our pointers to concentrate on this region
            else:
                r = mid - 1

        # right sorted portion
        else:
            if target < nums[mid] or target > nums[r]:
                r = mid - 1
            else:
                l = mid + 1
    # when the target is not in our list of numbers, we just return -1
    return -1


# =================================================================== #

### TWO POINTERS ###
# ---------- 12. Container With Most Water - Leetcode 11 - Medium -------------
def maxArea(height):
    # initialize pointers
    l = 0
    r = len(height) - 1
    # initialize variable that'll store the maximum volume
    res = 0

    # continue with the loop as long as the pointers do not cross each other
    while l < r:
        # calculate the current area at the specific point in the iteration
        # it is basic equation of base*height where the base is the difference
        # in the pointers and the height is the smaller of the 2 values at the
        # left and right pointers
        currArea = (r - l) * min(height[l], height[r])
        # the current maximum volume at the specific point in the iteration is
        # just the bigger of the previous volume and the current volume
        res = max(res, currArea)
        # when the height at the left pointer is smaller than the height at the
        # right pointer we increment the left pointer by one so as to still
        # preserve the bigger height at the right pointer since that height may
        # be the smaller of 2 heights later in the iteration
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1

    return res

# ---------- 30. Valid Palindrome - Leetcode 125 - Easy -------------


def isPalindrome(s):
    # rev stores the reverse of the input string
    rev = ""
    # sAlNum stores the alphanumeric characters of the input string
    sAlNum = ""
    for c in s:
        c = c.lower()
        if c.isalnum():
            rev = c + rev
            sAlNum += c
    # return whether the reverse of the alphanumeric characters of the input string
    # is equal to the alphanumeric characters of the input string
    return rev == sAlNum

# One could make their own alphanumeric function utilizing the ord function
# as so:


def alphanum(c):
    return (
        ord("A") <= ord(c) <= ord("Z")
        or ord("a") <= ord(c) <= ord("z")
        or ord("0") <= ord(c) <= ord("9")
    )

# a true two pointer solution


def isPalindrome(self, s: str) -> bool:
    l = 0
    r = len(s) - 1

    while l < r:
        # skip non-alphanumeric characters
        while l < r and not self.alphanum(s[l]):
            l += 1
        while l < r and not self.alphanum(s[r]):
            r -= 1
        # check whether the characters at the left and right pointers are equal
        # return False immediately if they are not
        if s[l].lower() != s[r].lower():
            return False
        l += 1
        r -= 1
    return True


# ---------- 39. 3Sum - Leetcode 15 - Medium -------------


def threeSum(nums):
    # sort the numbers
    nums.sort()
    # initialize the result array
    res = []

    # loop through the numbers
    for i in range(len(nums)):
        # when the number is greater than 0, we can safely break out of the loop
        # because we know that the numbers that follow will be GREATER THAN 0
        # and so the sum of 3 numbers will never be 0
        if nums[i] > 0:
            break
        # when the number is equal to the number before it, we can safely SKIP IT
        # because we've already considered it in a previous iteration
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # initialize the left and right pointers
        l = i + 1
        r = len(nums) - 1

        # continue with the loop as long as the pointers don't cross each other
        while l < r:
            # calculate the sum of the 3 numbers
            currSum = nums[i] + nums[l] + nums[r]
            # when the sum is less than 0, we increment the left pointer by one
            # so as to increase the sum
            if currSum < 0:
                l += 1
            # when the sum is greater than 0, we decrement the right pointer by
            # one so as to decrease the sum
            elif currSum > 0:
                r -= 1
            # when the sum is equal to 0, we append the 3 numbers to the result
            # array
            else:
                res.append([nums[i], nums[l], nums[r]])
                # we then increment the left pointer by one and decrement the
                # right pointer by one so as to look for other combinations
                # that sum up to 0
                l += 1
                r -= 1
                # when the number at the left pointer is equal to the number
                # before it, we increment the left pointer by one so as to skip it
                # if we don't check whether the number at the left pointer is equal
                # to the number before it, we'll end up with duplicate doublets for
                # our mini 2Sum problem which we do not want since the question asks
                # for unique triplets
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                # when the number at the right pointer is equal to the number
                # before it, we decrement the right pointer by one so as to
                # skip it
                while l < r and nums[r] == nums[r + 1]:
                    r -= 1
    return res


# =================================================================== #

### STACK ###
# ---------- 13. Valid Parentheses - Leetcode 20 - Easy -------------
def isValid(s):
    # stack to store potentially matching open parentheses
    stack = []
    # dictionary with closing to open parentheses as key:val pairs
    closeToOpen = {')': '(', ']': '[', '}': '{'}

    for char in s:
        # when the character we're looking at is a closing parens
        if char in closeToOpen:
            # when the stack is non-empty and the open parens character at the top
            # of the stack matches the open parens counterpart of the closing parens
            # we're looking at, then it means that we have a matching pair of parens
            # and we can pop the open parens character at the top of the stack
            if len(stack) != 0 and stack[-1] == closeToOpen[char]:
                stack.pop()
            # when the stack is empty or if the open parens character at the top
            # of the stack does not match the open parens counterpart of the closing
            # parens we're looking at, then it means that the input string is not a
            # valid parens
            # in the case of the stack being empty, a sample input string would be
            # '(()))[]{}' whereby the time we get to the third closing ) parens, the
            # stack will be empty since 2 pops of ( will have been made in prior
            # iterations
            # in the case of the open parens character at the top of the stack not
            # matching the open parens counterpart of the closing parens we're looking
            # at, a sample string would be '[{]}' whereby the stack will be non-empty
            # but by the time we get to the third character, the closing parens ], the
            # character at the top of the stack will be the prior { which does not match
            # the open parens counterpart of ]
            else:
                return False
        # when the character we're looking at is an open parens, we add it to the stack
        # and it will be compared in a later iteration
        else:
            stack.append(char)
    # the input string will only be a valid string if by the end of iterating the whole
    # string, the stack should be empty which is just the technical way of saying we have
    # crossed out all matching parens and we have made sure they appear in order
    return len(stack) == 0


# another implementation using no hashmap
def isValid(s):
    stack = []
    for c in s:
        if c == '(' or c == '[' or c == '{':
            stack.append(c)
        elif c == ')' and len(stack) > 0 and stack[-1] == '(':
            stack.pop()
        elif c == ']' and len(stack) > 0 and stack[-1] == '[':
            stack.pop()
        elif c == '}' and len(stack) > 0 and stack[-1] == '{':
            stack.pop()
        else:
            return False
    return len(stack) == 0

# ======================================================================================== #

### TRIES ###
# --------- 15. Implement Trie (Prefix Tree) - Leetcode 208 - Medium ------------


class TrieNode:
    # initialize the trie node
    # you can't have a trie or solve a trie-related question without first having
    # a way to represent the nodes
    def __init__(self):
        # each node will have children but instead of initializing an array composed
        # of 26 characters of the English alphabet but a hashmap doing the same thing
        # will be easier
        self.children = {}
        # we need a way to determine when we've reached the end of a word
        # we can do this by initializing a boolean to False but we can set it to true
        # is a certain character is the end of the word
        # notice how we're not actually storing the character itself in the trie node,
        # that's gonna be implicit from the hashmap
        # so if we were adding a lowercase character 'a', we'd have children['a'] = TrieNode(),
        # this is how we're gonna be inserting a node
        self.endOfWord = False


class Trie:
    def __init__(self):
        """
        Initialize your data structure here
        """
        # we only really need the root trie node which is going to potentially lead to 26
        # other nodes each representative of the 26 letters of the English alphabet
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie
        """

        # initially start at the root
        curr = self.root

        # then we're gonna go character by character in the word
        for c in word:
            # we're basically going to check 2 things, if the character already exists or not
            # when the character is not in the hashmap yet, or the children of the node we're
            # looking at, then that means that it has not been inserted yet and so we're going
            # to create a trie node for this character
            if c not in curr.children:
                curr.children[c] = TrieNode()
            # update the current pointer by moving to the node represented by the character
            # we're looking at
            curr = curr.children[c]

        # we've finished iterating through each character of the word we want to insert
        # our current pointer points to the trie node representing the last character in the
        # inserted word
        # to signify that we're done we need to mark this last character as the end of the
        # inserted word
        curr.endOfWord = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie
        """

        # initially start at the root
        curr = self.root

        # search through the word character by character
        for c in word:
            # if we reach a point in our iteration where the character is not among the list
            # of children of the current node, then it means that there is no trie node
            # dedicated to the character we're looking at and thus we return false
            if c not in curr.children:
                return False
            # update the current pointer by moving to the node represented by the character
            # we're looking at
            curr = curr.children[c]

        # we've reached the end of the word we're searching and we know that we've reached the
        # end of the word if the character is the end of the word
        return curr.endOfWord

    def startsWith(self, prefix: str) -> bool:
        """
        Returns is there is any word in the trie that starts with the given prefix
        """

        # initially start at the root
        curr = self.root

        # search through the prefix character by character
        for c in prefix:
            # if the character we're looking at is not among the children nodes of the current
            # node, then it means there is no trie node dedicated to this character and hence
            # we return False
            if c not in curr.children:
                return False
            # update the current pointer by moving to the node represented by the character
            # we're looking at
            curr = curr.children[c]

        # at the end we've looked at each character in the prefix and we now know for sure that
        # there are a set of nodes with these set of characters from the prefix and hence we
        # return true
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

# --------- 16. Design Add and Search Words Data Structure - Leetcode 211 - Medium ------------

# TODO: NOT FULLY UNDERSTOOD


class TrieNode:
    # initialize the trie node
    # you can't have a trie or solve a trie-related question without first having
    # a way to represent the nodes
    def __init__(self):
        # each node will have children but instead of initializing an array composed
        # of 26 characters of the English alphabet but a hashmap doing the same thing
        # will be easier
        self.children = {}
        # we need a way to determine when we've reached the end of a word
        # we can do this by initializing a boolean to False but we can set it to true
        # is a certain character is the end of the word
        # notice how we're not actually storing the character itself in the trie node,
        # that's gonna be implicit from the hashmap
        # so if we were adding a lowercase character 'a', we'd have children['a'] = TrieNode(),
        # this is how we're gonna be inserting a node
        self.endOfWord = False


class WordDictionary:

    def __init__(self):
        """
        initialize your data structure here
        """
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        # initially start at the (empty) root
        curr = self.root

        # then we're gonna go character by character in the word
        for c in word:
            # we're basically going to check 2 things, if the character already exists or not
            # when the character is not in the hashmap yet, or the children of the node we're
            # looking at, then that means that it has not been inserted yet and so we're going
            # to create a trie node for this character
            if c not in curr.children:
                curr.children[c] = TrieNode()
            # update the current pointer by moving to the node represented by the character
            # we're looking at
            curr = curr.children[c]

        # we've finished iterating through each character of the word we want to insert
        # our current pointer points to the trie node representing the last character in the
        # inserted word
        # to signify that we're done we need to mark this last character as the end of the
        # inserted word
        curr.endOfWord = True

    def search(self, word: str) -> bool:
        # a depth-first search as a way to look at every possible (tree) path when we encounter a dot character
        def dfs(j, root):
            curr = root
            # go through every character in the word as a means of searching whether each character is in the trie
            for i in range(j, len(word)):
                c = word[i]
                # when the character is a dot
                if c == '.':
                    # since a dot acts as a wildcard, we want to investigate every possible child branch of the current node we are looking at which is done by looping over all children of the current node
                    # notice that we are looping over the values since they are the actual children
                    for child in curr.children.values():
                        # we want to know what's the remaining portion of the word that we're trying to match
                        # we also want to know what is the current node in our trie
                        # i + 1 because we are skipping the dot
                        if dfs(i + 1, child):
                            return True
                    return False
                # when the character we're looking at is a regular lowercase character
                else:
                    # if the character we're looking at is not among the children nodes of the current
                    # node, then it means there is no trie node dedicated to this character and hence
                    # we return False
                    if c not in curr.children:
                        return False
                    # update the current pointer by moving to the node represented by the character
                    # we're looking at
                    curr = curr.children[c]
            return curr.endOfWord

        return dfs(0, self.root)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)


# ========================================================================================== #

### 1-D DYNAMIC PROGRAMMING ###
# ------------------ 21. Climbing Stairs - Leetcode 70 - Easy ---------------------
def climbStairs(n):
    """
    This question can easily be solved using DP we can solve this using brute force utilizing DFS
    and a top-down approach meaning that we start from the bottom of the stairs and work our way
    up to the nth stair and in each path counting how many ways we arrive to the nth stair

    However,the time complexity of this approach is bad since it's O(2^n) where n is the number
    of stairs in this top down approach, we repetitively do the same calculation, eg, we answer the
    question "how many ways does it take to get to the nth stair from the second one?" twice to
    mitigate this, we can using a DP-bottom-up approach where we start with the easier questions
    (when n = 5, say) "how many ways does it take to get to the 5th stair from the fourth/fifth
    stair?

    Answer: 1" we can then work ourselve to the top (zeroth stair) by saving the previous subproblem
    solutions in their own variables which will take O(n) time complexity if you trace the outputs
    of one and two below, you'll find that this is essentially a Fibonacci sequence problem
    """

    # one represents the number of ways to get to the nth stair from the (n - 1)st stair via one step
    # remember, we are doing a bottom up approach as opposed to starting from the zeroth stair
    # and working our way up as we already know our 2 bases cases which are:
    # - the number of ways to get to the nth stair from the nth stair is n
    # - the number of ways to get to the nth stair from the (n - 1)st stair is n
    one = 1
    # two represents the number of ways to get to the nth stair from the (n - 2)nd stair via two steps
    two = 1
    # we start at the 3rd stair since we already know the number of ways to get to the 1st and
    # 2nd stairs
    for i in range(n - 1):
        # we need to save the number of ways to get to the 3rd stair in a temporary variable
        temp = one
        # the number of ways to get to the 3rd stair is the sum of the number of ways to get to
        one = one + two
        # the number of ways to get to the 2nd stair
        two = temp
    # the number of ways to get to the nth stair
    return one

# ------------------ 22. House Robber - Leetcode 198 - Medium ---------------------


def rob(nums):
    """
    https://youtu.be/73r3KWiEvyk?t=361
    """
    # initialize variables that'll store the bounty as we progress through nums
    # notice that rob2 will hold the maximum bounty as we progress through nums
    rob1 = 0
    rob2 = 0

    # [rob1, rob2, n, n + 1, ...]
    # essentially, if we are at value n, rob1 represents the result of robbing all
    # houses till the (n - 2)nd house but till the (n - 1)st house for rob2
    for n in nums:
        # the first argument entails if we decide to rob the house we're at in the
        # iteration and rob2 is if we decide not to rob the house
        temp = max(n + rob1, rob2)
        # our new rob1 will be where the prior rob2 was
        rob1 = rob2
        # our new rob2 will be the max just calculated
        # this means that rob2 signifies the maximum bounty we can get till the nth
        # house
        rob2 = temp
    # the maximum bounty we can get till the last house
    return rob2

# ------------------ 23. House Robber II - Leetcode 213 - Medium ---------------------


def rob(nums):
    """
    The added twist to this version of house robber is that the first and last houses
    are connected, i.e., the houses in the neighborhood are arranged in a circular shape
    therefore, we can't rob the first and last houses together without alerting the police
    one way to circumnavigate this extra inconvenience is to consider 3 scenarios:

    - calling the House Robber I code on nums less the first house, i.e., we don't rob the
    first house
    - calling the House Robber I code on nums less the last house, i.e., we don't rob the
    last house
    - an edge case materializes when we are only robbing one house, then the above 2 scenarios
    result in an empty nums list (thus a result of 0) when actually we can rob the sole house
    which would undoubtedly result in a bounty greater than 0 so we also consider nums[0] when
    taking the max of these 3 scenarios
    """
    # see 22: House Robber
    def rob1(nums):
        rob1 = 0
        rob2 = 0
        # [rob1, rob2, n, n+1, ...]
        for n in nums:
            temp = max(n + rob1, rob2)
            rob1 = rob2
            rob2 = temp
        return rob2
    return max(nums[0], rob1(nums[1:]), rob1(nums[0:len(nums) - 1]))

# ------------------ 24. Longest Palindromic Substring - Leetcode 5 - Medium ---------------------
# This is not a dynamic programming solution but a solution that utilizes 2 pointers


def longestPalindrome(s):
    # initialize the result
    res = ''
    # initialize the variable that'll hold the length of the longest palindrome
    resLen = 0

    # loop through the input string
    for i in range(len(s)):
        # odd length
        # we want to check the positions to the left and right of the position we're
        # currently at
        l, r = i, i
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > resLen:
                res = s[l:r + 1]
                resLen = r - l + 1
            l -= 1
            r += 1

        # even length
        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > resLen:
                res = s[l:r + 1]
                resLen = r - l + 1
            l -= 1
            r += 1

        return res


# ------------------ 50. Palindromic Substrings - Leetcode 647 - Medium ---------------------
def countSubstrings(s):
    # O(n^2) time complexity
    # the overall result
    res = 0

    # loop through the input string
    for i in range(len(s)):
        # odd length palindromes implementation
        l = i
        r = i
        # count the number of odd length palindromes
        res += countPalindromes(s, l, r)

        # even length palindromes implementation
        l = i
        r = i + 1
        # add the number of even length palindromes to the odd length palindromes already
        # calculated above
        res += countPalindromes(s, l, r)

    return res


def countPalindromes(s, l, r):
    """
    Helper function that counts the number of palindromes given the indices of the
    left and right pointers
    """
    # the individual result depending on whether we call this function on the odd or even
    # length implementation
    res = 0
    # starting from the middle and working our way outwards, check whether substring is
    # a palindrome
    while l >= 0 and r < len(s) and s[l] == s[r]:
        res += 1
        l -= 1
        r += 1
    return res

# ------------------ 51. Decode Ways - Leetcode 91 - Medium ---------------------

# recursive approach


def numDecodingsRecursive(s):
    # time complexity: O(2^n)
    # space complexity: O(n)
    # this is the base case when the input string is empty
    dp = {len(s): 1}

    # i represents the index of the input string we're currently at
    def dfs(i):
        # base case 2
        # we can return the number of ways to decode the input string when we've
        # reached the end of the input string or if i has already been calculated/cached in dp
        if i in dp:
            return dp[i]
        # base case 3
        # we cannot decode an input string that starts with a 0
        if s[i] == '0':
            return 0

        # recursive case
        # we want to check the number of ways to decode the input string when we
        # include the current character we're looking at and when we DON'T include
        # the current character we're looking at
        res = dfs(i + 1)
        # when we can include the current character we're looking at
        # we want to check if we can include the next character as well
        # digits ranging from 10 to 19 and 20 to 26 (which is why we're checking whether
        # s[i + 1] in '0123456') are valid

        # basically, we run dfs on i + 1 when we're checking a single character only
        # say, for instance, we're looking at s = '123', dfs(i + 1) essentially asks, looking at i = 0,
        # what are the number of ways to decode the input string '23'? which is 2
        # dfs(i + 2) essentially asks, looking at i = 0 (i.e., the number 12 given by the constraints
        # in the if statement below), what are the number of ways to decode the input string '3'? which is 1
        if (i + 1 < len(s) and (s[i] == '1' or (s[i] == '2' and s[i + 1] in '0123456'))):
            res += dfs(i + 2)

        # cache the result
        dp[i] = res
        return res

    return dfs(0)

# dynamic programming approach


def numDecodingsDynamic(s):
    dp = {len(s): 1}

    for i in range(len(s) - 1, -1, -1):
        if s[i] == '0':
            dp[i] = 0
        else:
            dp[i] = dp[i + 1]

        if (i + 1 < len(s) and (s[i] == '1' or (s[i] == '2' and s[i + 1] in '0123456'))):
            dp[i] += dp[i + 2]

    return dp[0]


# ------------------ 52. Maximum Product Subarray - Leetcode 152 - Medium ---------------------
def maxProduct(nums):
    # the gist of this problem is that we want to keep track of the maximum and minimum
    # products at each index of the input array
    # the reason why we want to keep track of the minimum product is because the minimum
    # product can become the maximum product when it is multiplied by a negative number
    # and vice versa
    # the reason why we want to keep track of the maximum product is because the maximum
    # product can become the minimum product when it is multiplied by a negative number
    # and vice versa
    # we want to keep track of the maximum product at each index because we want to
    # return the maximum product of the whole array

    # initialize the result - the reason why we're initializing the result to the maximum
    # value of the input array is because the input array could be all negative numbers
    # and so the maximum product would be the smallest negative number
    res = max(nums)
    # initialize the current minimum product and current maximum product to be 1
    currentMin = 1
    currentMax = 1

    # loop through the input array
    for n in nums:
        # the edge case is when the number we're looking at is 0
        # when the number we're looking at is 0, we want to reset the current minimum
        # and current maximum products to 1 because we want to start over
        # the reason why we're not resetting the res variable to 0 is because the res
        # variable could be the maximum value of the input array and so we don't want
        # to reset it to 0
        if n == 0:
            currentMin = 1
            currentMax = 1
            continue
        # store the current maximum product in a temporary variable because we want to
        # use it to calculate the current minimum product
        temp = currentMax
        # the current maximum product is the maximum of the following:
        # - the current number; if the current number is greater than the current maximum
        # product, then the current maximum product will be the current number
        # - the current number multiplied by the current maximum product
        # - the current number multiplied by the current minimum product; if the current
        # number is negative, then the current minimum product will be negative and so
        # multiplying the current number by the current minimum product will give us the
        # current maximum product
        currentMax = max(n * currentMax, n * currentMin, n)
        # the current minimum product is the minimum of the following:
        # - the current number; if the current number is less than the current minimum
        # product, then the current minimum product will be the current number
        # - the current number multiplied by the current maximum product; if the current
        # number is negative, then the current maximum product will be negative and so
        # multiplying the current number by the current maximum product will give us the
        # current minimum product
        currentMin = min(n * temp, n * currentMin, n)
        res = max(res, currentMax)

    return res

# ------------------ 53. Coin Change - Leetcode 322 - Medium ---------------------


def coinChange(coins, amount):
    # we want to keep track of the minimum number of coins needed to make up the amount
    # at each index of the dp array
    # the reason why we're initializing the dp array to amount + 1 is because the maximum
    # number of coins needed to make up the amount is amount

    # basically, we want to keep track of the minimum number of coins needed to make up
    # the amount at each index of the dp array and we are going to use the coins array
    # to calculate the minimum number of coins needed to make up the amount at each index
    # of the dp array via a bottom up approach
    dp = [amount + 1] * (amount + 1)

    # the minimum number of coins needed to make up the amount 0 is 0
    dp[0] = 0

    # loop through the dp array
    for a in range(1, amount + 1):
        # loop through the coins array for each amount
        for coin in coins:
            # when the coin is less than or equal to the amount
            # we want to check the minimum number of coins needed to make up the amount
            # at each index of the dp array
            # we need a - coin to be greater than or equal to 0 because we don't want to
            # be accessing negative indices of the dp array
            if a - coin >= 0:
                # the minimum number of coins needed to make up the amount at each index
                # of the dp array is the minimum of the following:
                # - the current minimum number of coins needed to make up the amount at
                # the current index of the dp array
                # - 1 + the minimum number of coins needed to make up the amount at the
                # index of the dp array that is the difference between the current amount
                # and the current coin
                dp[a] = min(dp[a], 1 + dp[a - coin])

    # when the amount is greater than the maximum number of coins needed to make up the
    # amount, then we return -1, it means that the amount cannot be made up by the coins
    # in the coins array and so we return -1
    if dp[amount] != amount + 1:
        return dp[amount]
    else:
        return -1


# =================================================================== #
### BACKTRACKING ###

# ------------------ 31. Combination Sum - Leetcode 39 - Medium ---------------------


def combinationSum(candidates, target):
    # global variable that'll store the result
    res = []
    # a Depth First Search that we'll use to traverse the state space tree
    # i maintains the candidates we're allowed to choose from; remember res sublists
    # like [2, 2, 3] and [2, 3, 2] are not allowed since they're essentially the same;
    # order matters
    # cur keeps track of the values we've currently added to the combination
    # we also want to be maintaining the total sum of the combination list because if
    # it ever goes over the target then we hit a base case and we stop

    def dfs(i, cur, total):
        # base case 1
        if total == target:
            # since we're only maintaining a single variable list for cur, we don't
            # want to actually append cur itself because we're going to continue to use
            # this cur variable when we're doing the other combinations recursively
            res.append(cur.copy())
            return
        if i >= len(candidates) or total > target:
            return

        # append the current candidate to cur
        cur.append(candidates[i])
        # call dfs on i and cur still but the total changes to included the newly
        # appended candidates[i]
        dfs(i, cur, total + candidates[i])

        # when we can't include the current candidate
        # we have to remove the previously added candidate
        cur.pop()
        dfs(i + 1, cur, total)

    dfs(0, [], 0)
    return res

# ------------------ 32. Word Search - Leetcode 79 - Medium ---------------------


def exist(board, word):
    # dimensions of the board
    rows = len(board)
    cols = len(board[0])
    # add all positions we've visited in the board to make sure we don't revisit them later
    path = set()

    # r, c - position in the board that we're at
    # i - current character within our target word we're looking for
    def dfs(r, c, i):
        # when we've traversed through the whole word, we know we've found the
        # word and we can return True - that the word exists in the board
        if i == len(word):
            return True
        # when we've gone out of bounds or when the current character we're looking
        # at is not the same as the character in the board, or if we've already visited
        # the position we're currently at, then we know that the word does not exist
        # we return False
        if (r < 0 or
            c < 0 or
            r >= rows or
            c >= cols or
            word[i] != board[r, c]
                or (r, c) in path):
            return False
        # add the position we're currently at to the path
        path.add((r, c))
        # we want to check the positions to the right, left, top, and bottom of the
        # position we're currently at so we run dfs on those positions
        res = (dfs(r + 1, c, i + 1) or
               dfs(r - 1, c, i + 1) or
               dfs(r, c + 1, i + 1) or
               dfs(r, c - 1, i + 1))
        # remove the position we're currently at from the path
        # the reason why we are removing the position we're currently at from the path
        # is because we are returning the result of the dfs calls above and if we don't
        # remove the position we're currently at from the path, then the dfs calls above
        # will always return true since the position we're currently at will always be
        # in the path
        path.remove(r, c)
        return res

    # brute force by going through every single position in the grid
    for r in range(rows):
        for c in range(cols):
            # if this returns true, we return true immediately, we only
            # need to find one instance of the word
            if dfs(r, c, 0):
                return True
    return False


# ==================================================================================== #
### INTERVALS ###
# ------------------ 33. Insert Interval - Leetcode 57 - Medium ---------------------
def insert(intervals, newInterval):
    # variable that'll store the result
    res = []
    # variable that'll store the index of the interval we're currently at
    for i in range(len(intervals)):
        # when the end of the new interval is less than the start of the current interval,
        # we add the new interval to the result and then add the rest of the intervals
        # because the new interval will not affect the rest of the intervals since the intervals
        # are sorted and are non-overlapping as indicated in the example [[1, 3], [6, 9]], for
        # instance, 3 < 6
        if newInterval[1] < intervals[i][0]:
            res.append(newInterval)
            return res + intervals[i:]
        # when the start of the new interval is greater than the end of the current interval, it
        # means that the new interval is completely to the right of the current interval and thus
        # we can add the current interval to the result
        elif newInterval[0] > intervals[i][1]:
            res.append(intervals[i])
        # when the start of the new interval is less than or equal to the end of the
        # current interval, we merge the intervals
        else:
            newInterval = [min(newInterval[0], intervals[i][0]),
                           max(newInterval[1], intervals[i][1])]
    # when we reach the end of the intervals list, we add the new interval to the result
    res.append(newInterval)
    return res

# ------------------ 34. Merge Intervals - Leetcode 56 - Medium ---------------------


def merge(intervals):
    # sort the intervals by their start times
    # intervals.sort() also works
    intervals.sort(key=lambda x: x[0])
    # variable that'll store the result
    res = []
    # add the first interval to the result to initiate the comparison process
    res.append(intervals[0])
    # variable that'll store the index of the interval we're currently at
    for i in range(len(intervals)):
        # when the end of the current interval is less than the start of the next interval,
        # we add the current interval to the result
        # this is because the intervals are supposed to be sorted and non-overlapping
        # in the case of [[1,2],[3,4]], for instance, 2 < 3
        if res[-1][1] < intervals[i][0]:
            res.append(intervals[i])
        # when the end of the current interval is greater than or equal to the start of the
        # next interval, we merge the intervals
        else:
            res[-1][1] = max(res[-1][1], intervals[i][1])
    # the result at the end will be the merged intervals
    return res

# ------------------ 35. Non-overlapping Intervals - Leetcode 435 - Medium ---------------------


def eraseOverlapIntervals(intervals):
    # sort the intervals by their start times
    intervals.sort(key=lambda x: x[0])
    # variable that'll store the result
    res = 0
    # variable that'll store the index of the previous interval we're currently at
    prevEnd = intervals[0][1]
    # variable that'll store the index of the interval we're currently at
    for start, end in intervals[1:]:
        # when the start of the current interval is greater than or equal to the end of the previous
        # interval's end, we update the previous interval's end to the current interval's end
        # this essentially means that there is no overlap between the intervals and we can safely set
        # the previous interval's end to the current interval's end for further comparison along the
        # rest of the intervals in the intervals list
        if start >= prevEnd:
            prevEnd = end
        # when the start of the current interval is less than the end of the previous interval's end,
        # we know that there's an overlap and we increment the result since at least one of the
        # overlapping intervals needs to go
        else:
            res += 1
            # we choose the prevEnd to be the interval that ends first since we want it is less likely
            # that that end will overlap with the next interval
            # in the case of [[1, 2], [1, 3], [2, 3], [3, 4]], the prevEnd will be 2 and not 3
            # by doing this we are eliminating the interval that ends later
            prevEnd = min(prevEnd, end)
    # the result at the end will be the number of intervals that need to be removed
    return res
