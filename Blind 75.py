### LINKED LISTS ###
import collections


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
    
    # Advances first pointer so that the gap between first and second is n nodes
    # apart
    for i in range(n + 1):
        first = first.next
        
    # While the first pointer does not equal null move both first and second to 
    # maintain the gap and get nth node from the end
    while (first != None):
        first = first.next
        second = second.next
    
    # Delete the node being pointed to by second.
    second.next = second.next.next
    
    # Return dummy.next
    return dummy.next

# --------- 14. Reverse Linked List - Leetcode 209 - Easy ------------
def reverseList(head):
    # intialize the 2 needed pointers required to traverse through the
    # linked list
    prev = None
    curr = head

    # continue with the loop as long as the current pointer does not
    # point at a null node
    while curr:
        # save the reference to the next node after the current one since
        # it the next iteration it will serve as the current node
        placeholder = curr.next
        # changing the "direction of the arrow" or where the current node
        # points to
        curr.next = prev
        # advancing the previous pointer to be where the current pointer is 
        prev = curr
        # advancing the current pointer to the placeholder we conviently
        # saved earlier
        curr = placeholder
    
    # the new head of the reversed linked list will be the node the prev
    # pointer is pointing to
    return prev

# ------------- 17. Merge Two Sorted Lists - Leetcode 21 - Easy -----------------
def mergeTwoLists(l1, l2):
    # create a temp head node which will form the basis of the result linked list
    res = ListNode()
    # a pointer to the temp head
    tail = res

    # while both pointers to the input linked lists are non-null, continue looping
    while l1 and l2:
        # if at a point in the iteration, the value at the l1 node is less than the
        # value at the l2 node, then we add the lesser l1 node value to our res
        # linked list
        if l1.val < l2.val:
            tail.next = l1
            # advanced the l1 pointer forward
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        # we advance the pointer to the result linked list regardless
        tail = tail.next

    # the not-so-happy case when the lengths of the input linked lists are unequal
    if l1:
        tail.next = l1
    else:
        tail.next = l2
    # since the default head node of our result list is 0, we need to return the
    # next node after the head
    return res.next

# ----------------- 18. Linked List Cycle - Leetcode 141 - Easy --------------------
def hasCycle(head):
    # using slow and fast pointers
    # the gist is that one pointer advances faster than the other and if the linked
    # list has a cycle, the fast one will eventually overlap the slow one and when
    # that happens is when we know the pointer has a cycle
    # we initially set the 2 pointers at the head of the input linked list
    slow = head
    fast = head
    
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
    Do not return anything, modify head in-place instead.
    """
    # find the middle point of the linked list
    # the gist of the implementation is that we need to split the input linked list
    # into 2 halves
    # since we're alternating, we need pointers to heads of the 2 halves, the only
    # challenge is that we need to reverse the second halve of the linked list for
    # easier reordering as we can't "go back" in a singly linked list
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
    # after the second while loop is done executing the head of the now reversed second
    # half of the linked list will be at prev as second(the pointer of the second half
    # of the linked list) will be at Null
    second = prev
    # we continue merging until one of the pointers, either first or second, is non-null
    # but since we know that the second half could be shorter, our condition could be
    # predicated on just the second pointer
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
        
    
# =================================================================== #

### TREES ###

class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
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

# ---------- 20. Maximum Depth of Binary Tree - Leetcode 104 - Easy -------------
def maxDepth(root):
    # the base case
    # when we reach a null node, we return 0 since the depth of a null node is 0
    if root is None:
        return 0

    # the recursive case
    # calculate the depth of the left part of the binary tree
    leftDepth = maxDepth(root.left)
    # calculate the depth of the right part of the binary tree
    rightDepth = maxDepth(root.right)

    # the maximum depth will be the larger of the depths of either the left or
    # right parts of the binary tree
    # plus one because of the fact that we do not initially consider the height
    # of the root
    return max(leftDepth, rightDepth) + 1

# --------------- 26. Same Tree - Leetcode 100 - Easy --------------
def isSameTree(p, q):
    # base case part 1
    # when both nodes are null, we can consider them to be the
    # same tree and hence we return true
    if p == None and q == None:
        return True
    
    # base case part 2
    # if one of the nodes is null and the other is not OR the nodes
    # we're comparing are non-null but don't have the same value, they
    # are not the same tree and so we return false
    if (p == None or q == None) or (p.val != q.val):
        return False
    
    # the recursive case
    # both the left and right sides of the tree need to be strictly equal,
    # that is, if, say, the left subtrees of p and q are equal while their
    # right subtrees are not, we do not consider them to be the same tree
    # and so their logical AND returns False
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
    
    # if the subroot is null, then technically it is a subtree of any other tree since
    # you can always find a null node in the leaf children of any other tree
    if not subroot:
        return True
    
    # if the root is null, then technically nothing, other than the null node (which
    # we checked above), can be the subtree of the null node
    if not root:
        return False
    
    if isSameTree(root, subroot):
        return True
    
    return (isSubtree(root.left, subroot) or
            isSubtree(root.right, subroot))

# --------------- 28. Lowest Common Ancestor of a Binary Search Tree - Leetcode 235 - Medium --------------
def lowestCommonAncestor(root, p, q):
    # we always start at the root because it is always going to be a common ancestor for every node
    cur = root
    # proceed with the loop as long as the node we're looking at is not null
    while cur:
        # if both p and q's values are greater than the root's value, we restrict our search to the right subtree
        if p.val > root.val and  q.val > root.val:
            cur = cur.right
        # if both p and q's values are less than the root's value, we restrict our search to the left subtree
        elif p.val < root.val and  q.val < root.val:
            cur = cur.left
        # it means that either p or q's value is equal to the root OR that a split occured and p is in the left
        # subtree and q is in the right subtree
        # in either case we just return cur, the current root node
        else:
            return cur

# --------------- 29. Binary Tree Level Order Traversal - Leetcode 102 - Medium --------------
def levelOrder(root):
    # the list that'll hold the result
    res = []
    # initialize queue for Breadth First Search (BFS)
    q = collections.deque()
    # initialize queue with the given root node
    q.append(root)

    # run BFS while queue is nonempty
    while q:
        # get number of nodes that are in the queue at a given point/currently
        # ensures that we go through the queue one level at a time
        qLen = len(q)
        level = []
        # loop through every value in the queue currently
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
        # append every single level to the result
        # make sure that the level list in non-empty
        if level:
            res.append(level)
    return res


# =================================================================== #

### GRAPHS ###
from collections import defaultdict, deque

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
            a breadth first search to check the number of islands by marking already visited islands so as
            to not forget which ones have already been visited
            """
            # bfs is an iterative algorithm that needs a DS, which is normally a queue
            q = deque()
            # we add the island to the visited pile
            visited.add((r, c))
            # append the island we're at in the iteration in the our bfs queue
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
                    if r in range(rows) and c in range(cols) and grid[r][c] == '1' and (r, c) not in visited:
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

# ---------------- 9. Two Sum - Leetcode 1 - Easy -------------------
def twoSum(nums, target):
    # dictionary that'll store already visited numbers in nums alongside 
    # their indices as key:value pairs. conveniently, the values will be
    # the index of each number
    twoSumDict = {}
    
    # looping through nums, where each number in each iteration will act
    # as num1
    for i in range(len(nums)):
        # the second number, num2, that when added to num1 will produce
        # target
        num2 = target - nums[i]
        # when num2 is already in the dictionary, it means we've already
        # found our 2 numbers and we can return their indices as a list
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
        self.small = [] # maxheap
        self.large = [] # minHeap (Python's default)

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
    # initialize pointer
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
        if prices[r] <= prices[l]:
            l = r
            r += 1
        else:
            # otherwise we calculate the current profit normally
            currProfit = prices[r] - prices[l]
            # the maximum profit will be the larger profit between the previous max
            # profit and the just calculated current profit
            maxProfit = max(maxProfit, currProfit)
            # we increment the right pointer by onw
            r += 1
    # the maximum profit at the end
    return maxProfit

# --------- 7. Longest Substring Without Repeating Characters - Leetcode 3 - Medium ------------
def lengthOfLongestSubstring(s):
    # create a set that'll store all unique non-repeating characters
    charSet = set()
    # initialize left pointer at first element of input string
    l = 0
    # result that'll hold the longest substring without repeating characters
    res = 0

    # loop through the input string
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
        # plus one since we're dealing with indices
        # the plus one is also because the left pointer is initially at the first element
        # of the input string
        res = max(res, r - l + 1)
    return res

# =================================================================== #

### BINARY SEARCH ###
# --------- 8. Find Minimum in Rotated Sorted Array - Leetcode 153 - Medium ------------
def findMin(nums):
    # variable that'll store the current minimum
    res = nums[0]
    # left pointer initially at the leftmost end
    l = 0
    # right pointer initially at the rightmost end
    r = len(nums) - 1

    # continue looping as long left and right pointers don't cross each other
    while l <= r:
        # if the number at the left pointer is less than the one at
        # the right pointer, it means that nums is already sorted and
        # we can safely return the number at the left pointer or the 
        # current minimum, whichever is smaller
        if nums[l] < nums[r]:
            res = min(res, nums[l])
            break

        # calculation of the location of the middle pointer
        mid = (l + r) // 2
        # before further comparison, the number at the middle pointer will serve as
        # the minimum
        res = min(res, nums[mid])
        # if the number at the middle is greater than or equal than the number at 
        # the left pointer, it means that we need to look at the right part of nums 
        # because it means that the left part of the sublist is already sorted and
        # because of the rotation, it makes no sense to look at the left part of
        # nums since it will always be larger than the right part of nums, so we
        # focus our attention to the right part of nums
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
        # if the number at the middle is less than the number at the left pointer, 
        # we are at the left sorted portion
        if nums[l] <= nums[mid]:
            # if the target greater than the number at the middle OR if the target is 
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
    # variable that'll store the reverse of input string
    rev = ""
    # adding the characters in s in reverse to rev
    for c in s:
        c = c.lower()
        if c.isalnum():
            rev = c + rev
    # variable that'll store input string but stripped off all
    # non-alphanumeric characters
    sAlNum = ""
    for c in s:
        c = c.lower()
        if c.isalnum():
            sAlNum += c
    # if palindrome, then will return true
    return rev == sAlNum

# One could make their own alphanumeric function utilizing the ord function
# as so:
def alphanum(c):
    return (
        ord("A") <= ord(c) <= ord("Z")
        or ord("a") <= ord(c) <= ord("z")
        or ord("0") <= ord(c) <= ord("9")
        )


# =================================================================== #

### STACK ###
# ---------- 13. Valid Parentheses - Leetcode 20 - Easy -------------
def isValid(s):
    # stack to store potentially matching open parens
    stack = []
    # dictionary with closing to open parentheses as key:val pairs
    closeToOpen = {')':'(', ']':'[', '}':'{'}

    # looping through the characters in string
    for char in s:
        # what to do when the character we're looking at is a closing parens
        if char in closeToOpen:
            # when the stack is non-empty and the last character in the stack 
            # or the character at the top of the stack (which is supposed to 
            # be an opening parens) matches the opening parens counterpart of
            # the closing parens we are looking at, then we remove the matching
            # opening parens at the top of the stack
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

#TODO: NOT FULLY UNDERSTOOD
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

    # one represents the number of ways to get to the nth stair from the (n - 1)st stair
    # remember, we are doing a bottom up approach as opposed to starting from the zeroth stair
    # and working our way up as we already know our 2 bases cases which are:
    # - the number of ways to get to the nth stair from the nth stair is n
    # - the number of ways to get to the nth stair from the (n - 1)st stair is n
    one = 1
    # two represents the number of ways to get to the nth stair from the (n - 1)st stair
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
def longestPalindrome(s):
    # initialize the result
    res = ''
    # initialize the variable that'll hold the length of the longest palindrome
    resLen = 0

    # loop through
    for i in range(len(s)):
        # odd length
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
    
# =================================================================== #
### BACKTRACKING ###

# ------------------ 31. Combination Sum - Leetcode 39 - Medium ---------------------
def combinationSum(candidates, target):
    # global variable that'll store the result
    res = []
    # a Depth First Search that we'll use to traverse the state space tree
    # i maintains the candidates we're allowed to choose from; remember res sublists
    # like [2,2,3] and [2,3,2] are not allowed since they're essentially the same;
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
            r>= rows or
            c >= cols or
            word[i] != board[r,c]
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
    intervals.sort(key = lambda x:x[0])
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
    intervals.sort(key = lambda x:x[0])
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