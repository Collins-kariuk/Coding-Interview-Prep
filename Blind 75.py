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

# --------------- 21. Same Tree - Leetcode 100 - Easy --------------
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


# =================================================================== #

### GRAPHS ###
from collections import deque

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

    # the right pointer that'll traverse/loop through the list as it checks for repeating
    # characters
    for r in range(len(s)):
        # when the character at the right pointer is in the set (meaning that character at that
        # point in the loop is repeated) we remove characters starting from the left of the 
        # substring with the repetition in it until that substring does not contain the current
        # character in s[r]
        while s[r] in charSet:
            # we have to remove it in 2 ways
            # 1. remove it from the set
            charSet.remove(s[l])
            # 2. advance the left pointer rightwards
            l += 1
        # if the character at s[r] is not in the set, it means we've not encountered a
        # repetition and we can add it to the set 
        charSet.add(s[r])
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
# --------- 21. Climbing Stairs - Leetcode 70 - Easy ------------
def climbStairs(n):
    """
    this question can easily be solved using DP
    we can solve this using brute force utilizing DFS and a top-down approach meaning that we
    start from the bottom of the stairs and work our way up to the nth stair and in each path
    counting how many ways we arrive to the nth stair
    however, the time complexity of this approach is bad since it's O(2^n) where n is the number
    of stairs
    in this top down approach, we repetitively do the same calculation, eg, we answer the
    question "how many ways does it take to get to the nth stair from the second one?" twice
    to mitigate this, we can using a DP-bottom-up approach where we start with the easier
    questions (when n=5, say) "how many ways does it take to get to the 5th stair from the
    fourth/fifth stair? Answer: 1"
    we can then work ourselve to the top (zeroth stair) by saving the previous subproblem solutions
    in their own variables which will take O(n) time complexity
    if you trace the outputs of one and two below, you'll find that this is essentially a Fibonacci
    sequence problem
    """

    # one and two represent the sort of base cases for this DP problem representing the number of
    # ways it takes to get to the nth stair from the (n-1)st and nth stair respectively
    one = 1
    two = 1

    # the bottom-up approach, working ourselves to the zero-th stair (from the nth stair)
    for i in range(n - 1):
        # well, since we definitively know the number of ways it takes to get to the nth stair from
        # the (n-1)st and nth stair respectively, we can use this to our advantage as we calculate
        # later number of ways
        # our new two will be where the old one was (so we save it in a variable) and our new one
        # will be whatever numbers of ways it took to get to the nth stair from the prior 2, one & two
        temp = one
        one = one + two
        two = temp
    # after all the iterations, where the last one lands will be the distinct number of ways we can
    # climb the stairs
    return one

# ------------------ 22. House Robber - Leetcode 198 - Medium ---------------------
def rob(nums):
    """
    https://youtu.be/73r3KWiEvyk?t=361
    """
    # we initialize the rob values to 0 in case nums is an empty list
    rob1 = 0
    rob2 = 0

    # [rob1, rob2, n, n+1, ...]
    # essentially, if we are at value n, rob1 represents the result of robbing all
    # houses till the (n-2)nd house but till the (n-1)st house for rob2
    for n in nums:
        # the first argument entails if we decide to rob the house we're at in the
        # iteration and rob2 is if we decide not to rob the house
        temp = max(n + rob1, rob2)
        # our new rob1 will be where the prior rob2 was
        rob1 = rob2
        # our new rob2 will be the max just calculated
        rob2 = temp
    
    return rob2