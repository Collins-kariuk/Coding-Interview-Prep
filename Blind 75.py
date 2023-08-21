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
    
# =================================================================== #

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
