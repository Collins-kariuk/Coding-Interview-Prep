# --------- 1. Merge Strings Alternately - Leetcode 1768 - Easy ------------
def mergeAlternately(word1, word2):
    l = 0
    r = 0
    res = []
    while l < len(word1) or r < len(word2):
        if l < len(word1):
            res.append(word1[l])
            l += 1
        if r < len(word2):
            res.append(word2[r])
            r += 1
    return ''.join(res)


# --------- 2. Greatest Common Divisor of Strings - Leetcode 1071 - Easy ------------
def gcdOfStrings(str1, str2):
    len1 = len(str1)
    len2 = len(str2)

    def isDivisor(l):
        if len1 % l != 0 or len2 % l != 0:
            return False
        factor1 = len1 // l
        factor2 = len2 // l
        return str1[:l] * factor1 == str1 and str1[:l] * factor2 == str2
    
    for l in range(min(len1, len2), 0, -1):
        if isDivisor(l):
            return str1[:l]
    return ""


# --------- 3. Delete the Middle Node of a Linked List - Leetcode 2095 - Medium ------------
def deleteMiddle(head):
    res = head
    traverser = head
    
    def counter(headNode):
        counter = 0
        counterPointer = head
        while counterPointer:
            counter += 1
            counterPointer = counterPointer.next
        return counter
    
    numNodes = counter(head)
    for i in range((numNodes // 2) - 1):
        traverser = traverser.next
    
    if numNodes > 1:
        traverser.next = traverser.next.next
        return res
    return None


# --------- 4. Container With Most Water - Leetcode 11 - Medium ------------
def maxArea(height):
    l = 0
    r = len(height) - 1
    maxVol = 0
    while l < r:
        minHeight = min(height[l], height[r])
        pointerDifference = r - l
        maxVol = max(maxVol, pointerDifference * minHeight)
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return maxVol


# --------- 5. Reverse Linked List - Leetcode 206 - Easy ------------
def reverseList(head):
    prev = None
    curr = head

    while curr:
        savedNextNode = curr.next
        curr.next = prev
        prev = curr
        curr = savedNextNode
    
    return prev

# --------- 6. Maximum Average Subarray I - Leetcode 643 - Easy ------------
def findMaxAverage(nums, k):
    currSum = maxSum = sum(nums[:k])

    for i in range(k, len(nums)):
        currSum += nums[i] - nums[i - k]
        maxSum = max(maxSum, currSum)

    return maxSum / k
