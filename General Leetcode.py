# --------- 1. Median of Two Sorted Arrays - Leetcode 4 - Hard ------------
# There are two sorted arrays nums1 and nums2 of size m and n respectively.
# Find the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).
def findMedianSortedArrays(nums1, nums2):
    A, B = nums1, nums2
    # A is the shorter array
    # B is the longer array
    total = len(nums1) + len(nums2)
    # total is the total length of the two arrays
    half = total // 2

    # if A is longer than B, swap them
    if len(B) < len(A):
        A, B = B, A
    
    # perform binary search on A
    l, r = 0, len(A) - 1
    while True:
        # i is the index of the middle element of A
        i = (l + r) // 2
        # j is the index of the middle element of B
        # partition A and B such that the number of elements in the left partition
        # is equal to the number of elements in the right partition
        j = half - i - 2

        # Aleft is the left element of A
        # Aright is the right element of A
        Aleft = A[i] if i >= 0 else float("-infinity")
        Aright = A[i + 1] if (i + 1) < len(A) else float("infinity")
        # Bleft is the left element of B
        # Bright is the right element of B
        Bleft = B[j] if j >= 0 else float("-infinity")
        Bright = B[j + 1] if (j + 1) < len(B) else float("infinity")

        # if the partition is correct, return the median
        if Aleft <= Bright and Bleft <= Aright:
            # if the total length is odd, return the middle element
            if total % 2:
                return min(Aright, Bright)
            # if the total length is even, return the average of the middle two elements
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        # if the partition is not correct, move the partition
        # in this case, the left partition of A is too big
        # decrease A's left partition
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1

# ---------- 2. Valid Parentheses - Leetcode 20 - Easy -------------
# Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
# determine if the input string is valid.
# An input string is valid if:
# 1. Open brackets must be closed by the same type of brackets.
# 2. Open brackets must be closed in the correct order.
# Note that an empty string is also considered valid.
def isValid(s):
    # stack is a list that stores the opening brackets
    stack = []
    # iterate through the string
    for c in s:
        # if the character is an opening bracket, append it to the stack
        if c == '(' or c == '[' or c == '{':
            stack.append(c)
        # if the character is a closing bracket, check if the stack is empty
        # if the stack is empty, return False
        # if the stack is not empty, check if the top of the stack is the corresponding opening bracket
        # if the top of the stack is not the corresponding opening bracket, return False
        # if the top of the stack is the corresponding opening bracket, pop the top of the stack
        elif c == ')' and len(stack) > 0 and stack[-1] == '(':
            stack.pop()
        elif c == ']' and len(stack) > 0 and stack[-1] == '[':
            stack.pop()
        elif c == '}' and len(stack) > 0 and stack[-1] == '{':
            stack.pop()
        else:
            return False
    # if the stack is empty, return True
    return len(stack) == 0

# ---------- 3. Valid Anagram - Leetcode 242 - Easy -------------
# Given two strings s and t , write a function to determine if t is an anagram of s.
def isAnagram(s, t):
    # the advantage of using a dictionary is that it is easy to keep track of the number of occurences of each character
    # moreover, the advantage of using one dictionary is the reduction of space complexity
    # create a dictionary that stores the number of occurences of each character in s
    dic = {}
    for c in s:
        if c in dic:
            dic[c] += 1
        else:
            dic[c] = 1
    # iterate through t
    # if the character is in the dictionary, decrease the number of occurences of the character by 1
    # if the character is not in the dictionary, return False
    for c in t:
        if c in dic:
            dic[c] -= 1
        else:
            return False
    # if the number of occurences of each character in the dictionary is 0, return True
    for key in dic:
        if dic[key] != 0:
            return False
    return True