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
        # i is the index of the MIDDLE ELEMENT OF A
        i = (l + r) // 2
        # j is the index of the MIDDLE ELEMENT OF B
        # partition A and B such that the number of elements in the left partition
        # is equal to the number of elements in the right partition
        j = half - i - 2

        # Aleft is the left element of A (the rightmost element on the left of the partition)
        # Aright is the right element of A (the leftmost element on the right of the partition)
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
        # in this case, the left partition of A is too big decrease A's left partition
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

# ---------- 4. Decode String - Leetcode 394 - Medium -------------


def decodeString(s):
    """
    Given an encoded string, return its decoded string.
    The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times.
    Note that k is guaranteed to be a positive integer.
    You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.
    """

    # stack is a list that stores the characters
    # the advantage of using a stack is that it is easy to keep track of the characters
    stack = []

    # iterate through the string
    for i in range(len(s)):
        # if the character is not ']', append it to the stack
        if s[i] != ']':
            stack.append(s[i])
        else:
            # if the character is ']', pop the stack until '[' is reached
            substring = ''
            # substring is the string inside the (potentially inner) square brackets
            while stack[-1] != '[':
                # append the popped character to the substring
                # notice that we don't need to reverse the substring because we are popping the stack
                # from the end to the beginning and we are appending the popped character to the beginning of the substring
                # by not doing substring += stack.pop()
                substring = stack.pop() + substring
            # pop '['
            stack.pop()

            # k is the number of times the substring is repeated
            k = ''
            # pop the stack until a non-digit character is reached
            while len(stack) > 0 and stack[-1].isdigit():
                # append the popped character to k
                k = stack.pop() + k
            # append the repeated substring to the stack
            stack.append(substring * int(k))
    # return the decoded string
    return ''.join(stack)


# ---------- 5. Add Two Numbers - Leetcode 2 - Medium -------------
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def addTwoNumbers(l1, l2):
    # dummy is a dummy node
    dummy = ListNode(0)
    # cur is a pointer that points to the current node
    cur = dummy
    carry = 0

    while l1 or l2 or carry > 0:
        if l1 is None:
            v1 = 0
        else:
            v1 = l1.val

        if l2 is None:
            v2 = 0
        else:
            v2 = l2.val

        # new digit
        val = v1 + v2 + carry
        carry = val // 10

        # update val
        val = val % 10

        cur.next = ListNode(val)

        # update pointers
        cur = cur.next
        if l1 is None:
            l1 = None
        else:
            l1 = l1.next

        if l2 is None:
            l2 = None
        else:
            l2 = l2.next

    return dummy.next


# ---------- 6. Roman to Integer - Leetcode 13 - Easy -------------
def romanToInt(s):
    # romanToInt is a dictionary that stores the integer value of each roman numeral
    romanToInt = {'I': 1, 'V': 5, 'X': 10,
                  'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    # totalInt is the total integer value of the roman numeral
    totalInt = 0
    # iterate through the string
    for i in range(len(s) - 1):
        # if the current roman numeral is smaller than the next roman numeral, subtract
        # the current roman numeral from the total integer value otherwise, add the current
        # roman numeral to the total integer value
        if i + 1 < len(s) and romanToInt[s[i]] < romanToInt[s[i + 1]]:
            totalInt -= romanToInt[s[i]]
        else:
            totalInt += romanToInt[s[i]]
    # add the last roman numeral to the total integer value
    # notice that we don't need to check if the last roman numeral is smaller than the
    # second last roman numeral because the for loop will stop at the second last roman numeral
    return totalInt + romanToInt[s[-1]]


# ---------- 7. Longest Common Prefix - Leetcode 14 - Easy -------------
def longestCommonPrefix(inputList):
    # if the list is empty, return an empty string
    res = ""
    # iterate through the characters of the first string
    # notice that we can iterate through the characters of the first string because
    # the longest common prefix of the list is the same as the longest common prefix of the first string
    # and the longest common prefix is limited to the length of the shortest string too
    for i in range(len(inputList[0])):
        # iterate through the strings in the list
        for string in inputList:
            # if the index is out of range or the character is not the same, return the result
            if i >= len(string) or string[i] != inputList[0][i]:
                return res
        # if the character is the same for all strings, append the character to the result
        res += inputList[0][i]
    # return the result
    return res


# ---------- 8. Letter Combinations of a Phone Number - Leetcode 17 - Medium -------------
def letterCombinations(digits):
    # initialise the results list
    res = []
    # create a map of the numbers 2-9 to the digits under them
    numberDigit = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno',
                   '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

    # we are going to use a backtracking algorithm. The parameter i is going to tell us which character in our digit
    # string we are currently at and curStr (current string) which refers to the current string we're building.
    def backtrack(i, curStr):
        # the base case. The length of our current string is supposed to be equal to the length of the digits string
        # and if it is we just append that string to the res list. Another example of the base case would be if i >=
        # len(digits) since curStr and digits are supposed to be of the same length.
        if len(curStr) == len(digits):
            res.append(curStr)
            return
        # loop through the corresponding character values of each digit in digits. Call backtrack once more,
        # this time adding the char to our current string and adding one to i to move forward to the next digit.
        # Adding 1 to i means that we don't create combinations of characters from the same digit like with '2',
        # 'aa'. Adding c to our current string ensures that the base case will eventually be reached too.
        for c in numberDigit[digits[i]]:
            backtrack(i + 1, curStr + c)

    # We need digits to be of non-zero length
    if len(digits) != 0:
        backtrack(0, "")
    # return the result
    return res

# ---------- 9. Two Sum II - Input Array is Sorted - Leetcode 167 - Medium -------------


def twoSum(numbers, target):
    l = 0
    r = len(numbers) - 1

    while l < r:
        currentSum = numbers[l] + numbers[r]
        if currentSum == target:
            return [l + 1, r + 1]
        elif currentSum < target:
            l += 1
        else:
            r -= 1


# ---------- 10. 4Sum - Leetcode 18 - Medium -------------
def fourSum(nums, target):
    # sort the array
    nums.sort()
    # initialise the result list
    res = []
    # initialise the quadruplets list which will store the current quadruplet
    quadruplets = []

    # kSum is a recursive function that finds the kSum of the array
    def kSum(k, start, target):
        """
        @param k: the number of elements in the sum -
        notice that k is updated in each recursive call as it's decreased by 1
        @param start: the starting index of the array
        @param target: the target sum -
        notice that the target sum is updated in each recursive call
        """

        # the recursive case
        if k != 2:
            # iterate through the array until len(nums) - k + 1
            # we do this to ensure that there are enough elements in the array to form a kSum
            # for instance, if k = 4, then when we find the first number (nums[i]) in the quadruplet,
            # we need to find three more numbers in the array to form a quadruplet
            for i in range(start, len(nums) - k + 1):
                # we don't want any duplicates in the quadruplets list so we skip the duplicates in the array
                # we also need to check that i > start because if i == start, then we are at the first index of the array
                # and we don't want to check if the previous element is a duplicate because there is no previous element
                if i > start and nums[i] == nums[i - 1]:
                    continue
                # append the current number to the quadruplets list
                quadruplets.append(nums[i])
                # recursively call kSum with k - 1, i + 1 and target - nums[i], effectively moving it down to the next
                # number in the array
                kSum(k - 1, i + 1, target - nums[i])
                # pop the last element of the quadruplets list because we want to find the next quadruplet with the first
                # digit in the quadruplet potentially being different that the digit at start, i.e., by popping we're
                # giving other digits in nums a chance to be the first digit in the quadruplet
                quadruplets.pop()

        # the base case
        l = start
        r = len(nums) - 1

        # we've figured out the first two terms of the quadruplet and now the kSum has hit the base case and
        # reduces to Two Sum II but with a slight modification
        while l < r:
            # if the current sum is equal to the target, append the two values at the two pointers to the
            # quadruplets list
            currentSum = nums[l] + nums[r]
            if currentSum == target:
                res.append(quadruplets[:] + [nums[l], nums[r]])
                # it does not matter which pointer we update, as long as we update one of them, it is fine
                # since we don't want to end up with the same doublet
                l += 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
            # if the current sum is less than the target, increase the left pointer since we want the sum to
            # be bigger and increase to target
            elif currentSum < target:
                l += 1
            # if the current sum is greater than the target, decrease the right pointer since we want the
            # sum to be smaller and decrease to target
            else:
                r -= 1
    # call kSum
    # initially start starts at the first index of the array (0)
    kSum(4, 0, target)
    return res
