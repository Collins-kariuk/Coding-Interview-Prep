#### ARRAYS AND HASHING #####

# --------- 1. Contains Duplicate - Leetcode 217 - Easy ------------
import math
import collections
from collections import defaultdict, deque
import heapq
from collections import defaultdict


def containsDuplicate(nums):
    """
    Complexity O(n):
    The complexity of the containsDuplicate function is O(n), where n is the length of the input list nums.
    The function uses the set data structure to remove duplicates from the input list nums.
    The set data structure has an average time complexity of O(1) for operations like adding elements and checking for membership.
    In the code, set(nums) creates a set from the list nums, which eliminates any duplicate elements.
    The len(set(nums)) expression calculates the length of the set, which gives us the number of unique elements in nums.
    Finally, the function compares the length of the set with the length of the original list nums using the != operator.
    If the lengths are not equal, it means that there are duplicate elements in nums, and the function returns True. Otherwise, it returns False.
    Since creating the set and comparing the lengths both have a time complexity of O(n), the overall complexity of the function is O(n).
    """
    return len(set(nums)) != len(nums)
    # Complexity? O(1) space complexity and O(nlogn) time complexity

    # neetcode's solution - O(n) time and space complexity
    # hashset = set()
    # for n in nums:
    # if n in hashSet:
    #    return True
    # hashset.add(n)
    # return False


# --------- 2. Valid Anagram - Leetcode 242 - Easy ------------
def isAnagram(s, t):
    """
    Complexity O(n):
    The SPACE complexity of the isAnagram function is O(n), where n is the length of the input strings s and t.
    This is because the function creates two dictionaries, sDict and tDict, which store the frequency of each character in the respective strings.
    The size of these dictionaries depends on the number of unique characters in the strings.

    The TIME complexity of the isAnagram function is O(n), where n is the length of the input strings s and t.
    This is because the function iterates over each character in the strings to populate the dictionaries sDict and tDict.
    The iteration takes O(n) time, and the dictionary operations (checking for membership and updating values) have an average time complexity of O(1).
    Therefore, the overall space complexity and time complexity of the isAnagram function are both O(n).
    """

    def convertToDict(someString):
        stringDict = {}
        for c in someString:
            if c in stringDict:
                stringDict[c] += 1
            else:
                stringDict[c] = 1
        return stringDict

    sDict = convertToDict(s)
    tDict = convertToDict(t)

    return sDict == tDict


# --------- 3. Two Sum - Leetcode 1 - Easy ------------
def twoSum(nums, target):
    """
    COMPLEXITY:
    The space complexity of the twoSum function is O(n), where n is the length of the input list nums.
    This is because the function uses a dictionary, twoSumDict, to store the elements of nums as keys and their corresponding indices as values.
    The size of the dictionary depends on the number of unique elements in nums, which can be at most n.

    The time complexity of the twoSum function is O(n), where n is the length of the input list nums.
    This is because the function iterates over each element in nums once, performing constant-time operations such as dictionary lookups and updates.
    The worst-case scenario occurs when the target sum is achieved with the last two elements of nums, resulting in a linear time complexity.

    Therefore, the space complexity and time complexity of the twoSum function are both O(n).
    """

    twoSumDict = {}
    for i in range(len(nums)):
        num2 = target - nums[i]
        if num2 in twoSumDict:
            return [i, twoSumDict[num2]]
        else:
            twoSumDict[nums[i]] = i


# --------- 4. Group Anagrams - Leetcode 49 - Medium ------------
def groupAnagrams(strs):
    """
    COMPLEXITY:
    The space complexity of the groupAnagrams function is O(n), where n is the length of the input list strs.
    This is because the function uses a dictionary, anagramDict, to store the anagrams as keys and their corresponding indices as values.
    The size of the dictionary depends on the number of unique anagrams in strs, which can be at most n.

    The time complexity of the sorted function is O(n log n), where n is the length of the input string s.
    This is because the sorted function uses a SORTING algorithm that has a time complexity of O(n log n) in the average case.
    Therefore, the overall time complexity of the groupAnagrams function is O(n * m log m), where n is the LENGTH OF THE INPUT LIST strs and
    m is the MAXIMUM LENGTH OF A STRING in strs.
    This is because the function iterates over each element in strs and performs the sorted function on each string, which has a time complexity
    of O(m log m).
    """

    # The purpose of the anagramDict dictionary is to store anagrams.
    # The dictionary is used to store words as KEYS and their corresponding anagrams as VALUES.
    # This allows for efficient lookup and retrieval of anagrams based on a given word.
    anagramDict = {}
    for s in strs:
        # sorts the string
        # you first need to convert the string to a list of characters, sort the list, and then convert the sorted list back to a string.
        sortedS = "".join(sorted(s))

        # If sortedS is already a key in anagramDict, the code appends the value of another variable s to the list associated with that key.
        # If sortedS is not a key in anagramDict, the code creates a new key-value pair in anagramDict, where the key is sortedS and the value is a list containing only the value of s.
        if sortedS in anagramDict:
            anagramDict[sortedS].append(s)
        else:
            anagramDict[sortedS] = [s]
    # returns a list of all the values in anagramDict.
    return list(anagramDict.values())


# --------- 5. Top K Frequent Elements - Leetcode 347 - Medium ------------
def topKFrequent(nums, k):
    """
    COMPLEXITY:
    The space complexity of the topKFrequent function is O(n), where n is the length of the input list nums.
    This is because the function uses a dictionary, numDict, to store the elements of nums as keys and their corresponding frequencies as values.
    The size of the dictionary depends on the number of unique elements in nums, which can be at most n.

    The time complexity of the topKFrequent function is O(n), where n is the length of the input list nums.
    This is because the function iterates over each element in nums once, performing constant-time operations such as dictionary lookups and updates.
    The worst-case scenario occurs when all the elements in nums are unique, resulting in a linear time complexity.

    Therefore, the space complexity and time complexity of the topKFrequent function are both O(n).
    """
    numDict = {}
    for num in nums:
        if num in numDict:
            numDict[num] += 1
        else:
            numDict[num] = 1

    # a modified version of bucket sort
    frequencies = [[] for i in range(len(nums) + 1)]

    # frequencies is a list of lists, where each inner list contains all the numbers that occur with a certain frequency.
    # For example, if the number 2 occurs 3 times in nums, then frequencies[3] will contain the number 2.
    for number, frequencyOfNumber in numDict.items():
        frequencies[frequencyOfNumber].append(number)

    # The code iterates over the list frequencies in reverse order, starting from the last index.
    # This ensures that the numbers with the highest frequencies are added to the result list first.
    # The code then iterates over each number in the inner list and appends it to the result list.
    # The code returns the result list once it has k elements.
    res = []
    for i in range(len(frequencies) - 1, 0, -1):
        for num in frequencies[i]:
            res.append(num)
            if len(res) == k:
                return res


# --------- 6. Product of Array Except Self - Leetcode 238 - Medium ------------
def productExceptSelf(nums):
    """
    COMPLEXITY:

    The SPACE complexity is O(n), where n is the length of the input list nums.
    This is because the function uses two lists, leftProducts and rightProducts, to store the products of all the elements to the left and right of each element in nums.
    The size of these lists is equal to the length of nums, which is n.

    The TIME complexity of the productExceptSelf function is O(n), where n is the length of the input list nums.
    This is because the function iterates over each element in nums twice, performing constant-time operations such as list lookups and updates.
    Therefore, the space complexity and time complexity of the productExceptSelf function are both O(n).
    """

    # leftProducts[i] contains the product of all the elements to the left of nums[i] (excluding nums[i]).
    leftProducts = [1] * len(nums)
    for i in range(1, len(nums)):
        leftProducts[i] = leftProducts[i - 1] * nums[i - 1]

    # rightProducts[i] contains the product of all the elements to the right of nums[i] (excluding nums[i]).
    rightProducts = [1] * len(nums)
    for i in range(len(nums) - 2, -1, -1):
        rightProducts[i] = rightProducts[i + 1] * nums[i + 1]

    # The code iterates over each element in nums and multiplies the corresponding elements in leftProducts and rightProducts.
    # The code returns the result list once it has been populated.
    res = []
    for i in range(len(nums)):
        res.append(leftProducts[i] * rightProducts[i])
    return res


# solution I came up with
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
    print(" final pre", pre)
    # calculate the postfix array
    postMult = 1
    # we loop backwards
    for i in range(len(nums) - 1, 0, -1):
        postMult *= nums[i]
        post.append(postMult)
    print("post before reversing", post)
    # reverse the postfix array
    post = post[::-1]
    print("post after reversing", post)

    # first of all, initialize the result array to be all zeros because we
    # need to multiply the corresponding elements in the prefix and postfix
    # arrays and to index into the result array, we need to have something in
    # the result array so the zeros are the placeholders
    res = [0] * len(pre)
    for i in range(len(pre)):
        res[i] = pre[i] * post[i]

    return res


# --------- 7. Valid Sudoku - Leetcode 36 - Medium ------------
def isValidSudoku(board):
    """
    COMPLEXITY:

    The space complexity of the isValidSudoku function is O(1) because the space used by the cols, rows, and squares dictionaries
    is CONSTANT and does not depend on the size of the input.

    The time complexity of the isValidSudoku function is O(1) because the function iterates through a FIXED-SIZE 9x9 Sudoku board.
    The number of iterations is constant and does not depend on the size of the input. Therefore, the time complexity is constant.
    """

    # initialize the rows, cols, and squares dictionaries
    # the keys are the row/col/square numbers and the values are sets
    # the sets will contain the numbers that are in that particular row/col/square (will be populated as we iterate through the
    # board so we can check for duplicates in later iterations)
    # cols might look like {0: {5, 7}, 1: {1, 2, 3}, 2: {4, 6}}
    cols = collections.defaultdict(set)
    # rows might look like {0: {5, 7}, 1: {1, 2, 3}, 2: {4, 6}}
    rows = collections.defaultdict(set)
    # key = (r // 3, c // 3) so squares might look like {(0, 0): {5, 7}, (0, 1): {1, 2, 3}, (0, 2): {4, 6}}
    squares = collections.defaultdict(set)

    # iterate through the board
    for r in range(9):
        for c in range(9):
            # we don't care about the empty cells
            if board[r][c] == ".":
                continue
            # check if the number is already in the row/col/square
            # remember that the square is determined by the row and column number
            # for example, the square that contains the number in row 2 and column 3 is (2 // 3, 3 // 3) = (0, 1)
            if (board[r][c] in rows[r] or
                board[r][c] in cols[c] or
                    board[r][c] in squares[(r // 3, c // 3)]):
                return False
            # if the number is not in the row/col/square, add it to the respective set
            rows[r].add(board[r][c])
            cols[c].add(board[r][c])
            squares[(r // 3, c // 3)].add(board[r][c])
    return True


# --------- 8. Encode and Decode Strings - Lintcode 659 - Medium ------------
def encode(strs):
    """
    Encodes a list of strings to a single string.

    COMPLEXITY: 

    The res variable is initialized as an empty string ('').
    The encoded strings are concatenated to the res variable using the += operator.
    Since strings are immutable in Python, each concatenation operation creates a new string object.
    Therefore, the SPACE COMPLEXITY of the encode function is O(n), where n is the TOTAL LENGTH of all the input strings.

    The encode function iterates over each string in the strs list.
    For each string, it calculates the length using the len function, which has a TIME COMPLEXITY of O(1).
    It then performs string concatenation, which has a TIME COMPLEXITY of O(k), where k is the length of the current string.
    Since there are n STRINGS IN THE STRS LIST, the overall TIME COMPLEXITY of the encode function is O(n * k),
    where k is the AVERAGE LENGTH OF A STRING.
    """

    res = ''
    # the encoding of a certain word should be a string that starts with a number that
    # represents the length of the string followed by a hash sign then the word itself
    for string in strs:
        res += str(len(string)) + '#' + string
    return res


def decode(strs):
    """
    Decodes a single string to a list of strings.

    COMPLEXITY:

    The SPACE COMPLEXITY of the decode function is O(n), where n is the LENGTH OF THE INPUT STRING STRS.
    This is because the function creates a list res to store the decoded strings, and the size of this list will be proportional to the length of the input string.

    The TIME COMPLEXITY of the decode function is O(n), where n is the LENGTH OF THE INPUT STRING STRS.
    This is because the function iterates through the input string once, decoding each string and appending it to the result list.
    The length of the input string determines the number of iterations required, resulting in a linear time complexity.
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


# --------- 9. Longest Consecutive Sequence - Leetcode 128 - Medium ------------
def longestConsecutive(nums):
    """
    COMPLEXITY:

    The space complexity of the longestConsecutive function is O(n), where n is the LENGTH OF THE INPUT LIST nums.
    This is because the function creates a set numSet to store the unique elements of nums.
    The size of the set will be proportional to the length of the input list.

    The time complexity of the longestConsecutive function is O(n), where n is the LENGTH OF THE INPUT LIST nums.
    This is because the function iterates through the elements of nums once, performing constant-time operations for each element.
    The while loop inside the for loop MAY ITERATE MULTIPLE TIMES, but the total number of iterations is BOUNDED by the length of nums.
    Therefore, the time complexity is linear.
    """

    # convert to set to eliminate duplicates
    numSet = set(nums)
    longest = 0

    for n in nums:
        # Check whether the number before the current number is the start of a new sequence
        # if it is, then we can calculate the length of that particular sequence

        # We do this by checking whether the number before the current number is in the set
        # if it is, then we know that the current number is not the start of a new sequence
        # and we can safely skip it
        if n - 1 not in numSet:
            # to store the length of the current sequence
            curLength = 0

            # We calculate the length of the current sequence by checking whether the number
            # after the current number is in the set and if it is, we increment the length
            # Notice that since we initialize length to 0, we ensure that we count the
            # current start of the sequence as well
            while n + curLength in numSet:
                curLength += 1

            # we update the longest sequence if the current sequence is longer
            longest = max(curLength, longest)

    return longest


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### TWO POINTERS ####
# --------- 10. Valid Palindrome - Leetcode 125 - Easy ------------
def isPalindrome(s):
    """
    COMPLEXITY:

    The space complexity of the isPalindrome function is O(n), where n is the length of the input string s.
    This is because the function CREATES 2 LISTS, original and reversedOriginal, to store the alphanumeric characters of the string.
    The size of these lists will be proportional to the length of the input string.

    The time complexity of the isPalindrome function is O(n), where n is the length of the input string s.
    This is because the function ITERATES THROUGH each character of the string once to populate the original list.
    Additionally, the function creates the reversedOriginal list by REVERSING the original list, which takes O(n) time.
    Finally, the function compares the original and reversedOriginal lists, which also takes O(n) time.
    Therefore, the overall time complexity is linear.
    """

    # initialize two lists to store the original and reversed strings
    original = []
    reversedOriginal = []

    # populate the original list with only alphanumeric characters
    for c in s:
        # isalnum() checks whether the character is alphanumeric
        if c.isalnum():
            # lower() converts the character to lowercase
            original.append(c.lower())

    # populate the reversedOriginal list with only alphanumeric characters
    reversedOriginal = original[::-1]
    return original == reversedOriginal


# --------- 11. Two Sum II - Input Array Is Sorted - Leetcode 167 - Medium ------------
def twoSumSorted(numbers, target):
    """
    COMPLEXITY:

    The space complexity of the twoSumSorted function is O(1) because it uses a constant amount of extra space regardless of the input size.
    It only uses a few variables to store the pointers and the current sum.

    The time complexity of the twoSumSorted function is O(n), where n is the length of the numbers list.
    This is because the function uses a TWO-POINTER APPROACH to iterate through the list once.
    The pointers move towards each other until they meet, and at each iteration, the function compares the current sum with the target value.
    Since the list is sorted, the function can determine whether to move the left pointer or the right pointer based on the comparison.
    Therefore, the function performs a constant amount of work for each element in the list, resulting in a linear time complexity.
    """

    # initialize two pointers, one at the start and one at the end
    l = 0
    r = len(numbers) - 1

    # iterate until the pointers meet
    while l < r:
        # calculate the current sum
        currSum = numbers[l] + numbers[r]
        # if the current sum is greater than the target, we need to decrease the sum so we move the right pointer to the left
        if currSum > target:
            r -= 1
        elif currSum < target:
            l += 1
        # if the current sum is equal to the target, we return the indices (1-indexed)
        else:
            return [l + 1, r + 1]


# --------- 12. 3Sum - Leetcode 15 - Medium ------------
def threeSum(nums):
    """
    COMPLEXITY:

    The space complexity of the threeSum function is O(n), where n is the length of the nums list.
    This is because the function uses additional space to store the result array res, which can potentially contain all possible unique triplets that sum up to 0.
    In the worst case scenario, the size of res can be O(n^2) if all elements in nums form unique triplets.
    nums = [-1, -2, -3, ..., -n, 1, 2, 3, ..., n]

    The time complexity of the threeSum function is O(n^2).
    The function first SORTS the nums list, which takes O(n log n) time.
    Then, it iterates through each element in the sorted list, resulting in O(n) iterations.
    Within each iteration, the function uses a two-pointer approach to find the remaining two numbers that sum up to the negation of the current number.
    This TWO-POINTER APPROACH takes O(n) time in the worst case scenario, as the pointers can potentially traverse the entire list.
    Therefore, the overall time complexity is O(n log n + n^2), which SIMPLIFIES to O(n^2).
    """

    nums.sort()
    res = []

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
                # we change the states of the pointers by one so as to look for
                # other combinations that sum up to 0
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


# --------- 13. Container with Most Water - Leetcode 11 - Medium ------------
def maxArea(height):
    """
    COMPLEXITY:

    The time complexity of the maxArea function is O(n), where n is the length of the height list.
    This is because the function uses a TWO-POINTER approach to iterate through the list once.

    The space complexity of the maxArea function is O(1), as it only uses a constant amount of extra space to store the pointers and the result variable.
    """

    # initialize pointers
    l = 0
    r = len(height) - 1
    res = 0

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


# --------- 14. Trapping Rain Water - Leetcode 42 - Hard ------------
def trap(height):
    """
    COMPLEXITY:

    The time complexity of the trap function is O(n), where n is the length of the height list.
    This is because the function uses a TWO-POINTER approach to iterate through the list once.

    The space complexity of the trap function is O(1), as it only uses a constant amount of
    extra space to store the pointers, the left and right maxes, and the result variable.
    """

    # edge case when the height list is empty
    if len(height) == 0:
        return 0

    # initialize the left and right pointers
    l = 0
    r = len(height) - 1

    # initialize the left and right maxes
    leftMax = height[l]
    rightMax = height[r]

    # initialize the result variable
    res = 0

    while l < r:
        # If the left max is less than the right max, we know that the water
        # trapped at the left pointer is determined by the left max
        # We also know that the water trapped at the left pointer is determined
        # by the height at the left pointer
        # We can then calculate the water trapped at the left pointer by
        # subtracting the height at the left pointer from the left max
        # We then increment the left pointer by one so as to move on to the next
        # pointer
        if leftMax < rightMax:
            res += leftMax - height[l]
            l += 1
            # we update the left max if the height at the left pointer is greater
            # than the left max
            leftMax = max(leftMax, height[l])

        # if the right max is less than the left max, we know that the water
        # trapped at the right pointer is determined by the right max
        # we also know that the water trapped at the right pointer is determined
        # by the height at the right pointer
        # we can then calculate the water trapped at the right pointer by
        # subtracting the height at the right pointer from the right max
        # we then decrement the right pointer by one so as to move on to the next
        # pointer
        else:
            res += rightMax - height[r]
            r -= 1
            # we update the right max if the height at the right pointer is greater
            # than the right max
            rightMax = max(rightMax, height[r])

    return res


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### STACK ####
# --------- 15. Valid Parentheses - Leetcode 20 - Easy ------------
def isValid(s):
    """
    COMPLEXITY:

    The space complexity of the isValid function is O(n), where n is the length of the input string s.
    This is because the function uses a stack to store opening brackets, and the maximum size of the
    stack is proportional to the number of opening brackets in the string.

    The time complexity of the isValid function is O(n), where n is the length of the input string s.
    This is because the function iterates through each character in the string once, performing
    constant-time operations for each character.
    """

    # initialize a stack
    stack = []
    # a dictionary that maps the closing bracket to the opening bracket
    closeToOpen = {"]": "[", "}": "{", ")": "("}

    for c in s:
        # if the character is a closing bracket
        if c in closeToOpen:
            # if the stack is not empty and the top of the stack is the opening bracket
            # of the current closing bracket, we pop the opening bracket from the stack
            # else, we return False
            if len(stack) != 0 and stack[-1] == closeToOpen[c]:
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
        # if the character is an opening bracket, we push it onto the stack
        else:
            stack.append(c)
    # if the stack is empty, it means that all the opening brackets have been popped
    # and there are no closing brackets left and so the input string is valid
    return len(stack) == 0

# --------- 16. Min Stack - Leetcode 155 - Medium ------------


class MinStack:
    def __init__(self):
        # initialize the stack and the minStack
        # the stack will store the values while the minStack will store the minimum values after each push
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        # we push the value onto the stack
        self.stack.append(val)
        # we push the minimum value onto the minStack
        # we get the minimum value by comparing the current value with the value at the top of the minStack
        # if the minStack is empty, we push the current value onto the minStack which is just the equivalent of
        # pushing the current value onto the stack
        val = min(val, self.minStack[-1] if len(self.minStack) != 0 else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        # we return the value at the top of the stack
        return self.stack[-1]

    def getMin(self) -> int:
        # we return the value at the top of the minStack
        return self.minStack[-1]

# --------- 17. Evaluate Reverse Polish Situation - Leetcode 150 - Medium ------------


def evalRPN(tokens):
    """
    COMPLEXITY:

    The space complexity of the evalRPN function is O(n), where n is the number of tokens in the input list.
    This is because the function uses a stack to store the operands, and the size of the stack GROWS LINEARLY with the number of tokens.

    The time complexity of the evalRPN function is also O(n), where n is the number of tokens in the input list.
    This is because the function iterates through each token once and performs constant-time operations for each token.
    Therefore, the time complexity is linear with respect to the NUMBER OF TOKENS.
    """

    # Initialize a stack to store the operands
    stack = []

    for token in tokens:
        # Push the operands onto the stack
        if token not in "+-*/":
            stack.append(int(token))
        else:
            # Pop the operands in the correct order
            num2 = stack.pop()
            num1 = stack.pop()

            if token == "+":
                stack.append(num1 + num2)
            elif token == "-":
                stack.append(num1 - num2)
            elif token == "*":
                stack.append(num1 * num2)
            elif token == "/":
                # use integer division that truncates towards zero
                stack.append(int(num1 / num2))
    return stack[0]

# --------- 18. Generate Parentheses - Leetcode 22 - Medium ------------


def generateParenthesis(n):
    """
    COMPLEXITY:

    The space complexity of the generateParenthesis function is O(n), where n is the input parameter representing the number of pairs of parentheses.
    This is because the function uses a stack to store the parentheses combinations, and the maximum size of the stack at any given time is n.

    The stack indeed stores one potential parentheses combination at a time, but the maximum length of this combination can be up to 2n (for n pairs of parentheses,
    each pair consists of an opening and a closing bracket). Therefore, the space complexity is O(n) because the maximum size of the stack (or the maximum length of
    a single combination) is proportional to the input size n.
    Additionally, the recursive nature of the function also contributes to the space complexity. Each recursive call to backtrack adds a new level to the call stack.
    In the worst-case scenario, the depth of recursion (i.e., the maximum height of the implicit call stack) can be up to 2n, which also contributes to the O(n)
    space complexity. So, in summary, the space complexity of the function is O(n) due to the MAXIMUM SIZE OF THE STACK used to store a single parentheses
    combination and the MAXIMUM DEPTH OF THE RECURSIVE CALL STACK.

    The time complexity of the generateParenthesis function is O(4^n/n^(1/2)), which can be approximated as O(4^n). This is because the function uses
    a backtracking approach to generate all the valid parentheses combinations. In the worst case, the function explores all possible combinations,
    which is exponential with respect to the input size. The factor of 4 comes from the two choices for each parentheses position (open or closed),
    and the division by n^(1/2) accounts for the number of invalid combinations that are pruned during the backtracking process.
    """

    # initialize a stack to store the parentheses intermediately
    # at any given time in the backtracking process, the stack will store only one valid parentheses combination
    stack = []
    # initialize a list to store the result
    res = []

    def backtrack(openCount, closedCount):
        """
        The purpose of the backtrack function is to generate all the valid parentheses combinations.
        The function uses a backtracking approach to generate all the combinations.
        The function takes in two parameters, openCount and closedCount, which represent the number of open and closed parentheses respectively.

        Yes, that is correct. At any given time during the backtracking process, the stack will only be storing one potential parentheses combination.
        This is because the function uses a backtracking approach, where it explores all possible combinations by adding and removing parentheses from the stack.
        Each time a valid combination is found, it is appended to the result list, and then the function continues exploring other possibilities.
        """

        # the base case is when the number of open and closed parentheses is equal to n
        # this means that we have generated A valid parentheses combination
        # we append the parentheses combination to the result list and return
        if openCount == closedCount == n:
            res.append("".join(stack))
            return

        # as long as the number of open parentheses is less than n, we can add an open parentheses
        # to the stack
        if openCount < n:
            stack.append("(")
            # we increment the openCount by one since we have added an open parentheses
            # we keep the closedCount the same since we have not added a closed parentheses
            # this is done so that future recursive calls can keep track of the number of open and closed parentheses
            backtrack(openCount + 1, closedCount)
            stack.pop()

        if closedCount < openCount:
            stack.append(")")
            backtrack(openCount, closedCount + 1)
            stack.pop()

    # we initially call the backtrack function with the initial openCount and closedCount values of 0
    backtrack(0, 0)
    return res

# ------------- 19. Daily Temperatures - Leetcode 739 - Medium ---------------


def dailyTemperatures(temperatures):
    """
    COMPLEXITY:

    The time complexity of the dailyTemperatures function is O(n), where n is the length of the temperatures list.
    This is because we iterate through each temperature once and perform CONSTANT-TIME OPERATIONS for each temperature.

    The space complexity of the function is O(n) as well.
    This is because we use a stack to store the indices of the temperatures that we have seen so far alongside the temperature.
    The size of the stack grows linearly with the number of temperatures, so the space complexity is also linear with respect
    to the length of the temperatures list.
    """

    # initialize the result array
    # we initialize it to all zeros because the default value for when we do not find a warmer day is 0
    res = [0] * len(temperatures)
    # initialize a stack to store the indices of the temperatures that we have seen so far alongside the temperature
    stack = []  # pair: [temp, index]

    for currIndex, temperature in enumerate(temperatures):
        # if the stack is not empty and the current temperature is greater than the temperature at the top of the stack
        # we pop the temperature and index from the stack and calculate the number of days between the current index and
        # the index at the top of the stack
        # we then update the result array at the index at the top of the stack with the number of days
        # we continue popping from the stack until the stack is empty or the current temperature is less than the
        # temperature at the top of the stack
        while len(stack) != 0 and temperature > stack[-1][0]:
            stackTemp, stackInd = stack.pop()
            res[stackInd] = currIndex - stackInd
        # we push the current temperature and index onto the stack
        stack.append([temperature, currIndex])
    return res

# ------------- 20. Car Fleet - Leetcode 853 - Medium ---------------


def carFleet(target, position, speed):
    """
    COMPLEXITY:

    The time complexity of the carFleet function is O(n log n), where n is the length of the position or speed list.
    This is because the function involves SORTING the pair list, which takes O(n log n) time complexity.

    The space complexity of the function is O(n), where n is the length of the position or speed list. This is
    because the function creates A PAIR LIST to store the position and speed of each car, which takes O(n) space.
    Additionally, the function uses a stack to keep track of the time it takes for the cars to reach the target,
    which also takes O(n) space in the worst case.
    """

    # keep track of the position and speed of each car simultaneously in one list that'll store the pair as tuples
    # pair = [(p, s) for p, s in zip(position, speed)]
    # we could also do the above using the zip function but it's more readable to do it the way below
    pair = []
    for i in range(len(position)):
        pair.append((position[i], speed[i]))

    # sort the list in reverse order based on the position of the car because we want to start from the car that is
    # closest to the target
    pair.sort(reverse=True)
    stack = []
    # we use the stack to keep track of the time it takes for the cars to reach the target
    for p, s in pair:  # in reverse Sorted Order
        stack.append((target - p) / s)
        # as long as the stack has more than 1 car, we check whether the time it takes for the car at the top of the
        # stack to reach the target
        # is less than or equal to the time it takes for the car below it to reach the target
        # if it is, it means that the car below it will reach the target earlier or at the same time as the car above
        # it and so they will form a fleet and so we pop the car above it
        if len(stack) >= 2 and stack[-1] <= stack[-2]:
            stack.pop()
    # at the end of it all, the number of cars in the stack will be the number of fleets
    return len(stack)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### BINARY SEARCH ####
# ------------- 21. Binary Search - Leetcode 704 - Easy ---------------
def search(nums, target):
    """
    COMPLEXITY:

    The time complexity of the search function is O(log n), where n is the length of the nums list.
    This is because the function uses a BINARY SEARCH approach to find the target value.
    The function performs a constant amount of work for each iteration, and the number of iterations
    is bounded by the length of the nums list. Therefore, the time complexity is logarithmic.

    The space complexity of the search function is O(1), as it only uses a constant amount of extra
    space to store the pointers.
    """

    # initialize the left and right pointers
    l = 0
    r = len(nums) - 1

    # iterate until the pointers meet
    while l <= r:
        # calculate the middle index
        mid = (l + r) // 2
        # if the middle value is equal to the target, we return the middle index
        if nums[mid] == target:
            return mid
        # if the middle value is greater than the target, we move the right pointer to the left
        # so as to decrease the middle value
        elif nums[mid] > target:
            r = mid - 1
        # if the middle value is less than the target, we move the left pointer to the right
        # so as to increase the middle value
        else:
            l = mid + 1
    # if the pointers meet and we still haven't found the target, it means that the target is not
    # in the list and so we return -1
    return -1


# ------------- 22. Search a 2D Matrix - Leetcode 74 - Medium ---------------
def searchMatrix(matrix, target):
    """
    COMPLEXITY:

    The time complexity of the searchMatrix function is O(m + log n), where m is the number of rows and n is the number of columns.
    This is because the function uses a TWO-POINTER APPROACH to iterate through the matrix once.
    The function performs a constant amount of work for each iteration, and the number of iterations is bounded by the number of rows.
    Therefore, the time complexity is linear with respect to the number of rows.

    Additionally, the function uses a BINARY SEARCH approach to find the target value in the row.
    The function performs a constant amount of work for each iteration, and the number of iterations is bounded by the number of columns.
    Therefore, the time complexity is logarithmic with respect to the number of columns.

    The time complexity of the searchMatrix function is O(m + log(n)) instead of O(log(m*n)) because the function iterates through the
    matrix row by row using a two-pointer approach, and performs a binary search within each row.
    Iterating through the matrix: The function uses a for loop to iterate through each row of the matrix. Since there are m rows, the
    time complexity of this part is O(m).

    Binary search within each row: For each row, the function performs a binary search to find the target value. The binary search algorithm
    has a time complexity of O(log(n)), where n is the number of columns in the matrix. Since the binary search is performed for each row,
    the total time complexity for this part is O(m * log(n)).

    Combining the time complexities of both parts, we get O(m + m * log(n)), which can be simplified to O(m + log(n)).

    It's important to note that the time complexity is not O(log(m*n)) because the function does not perform a binary search on the entire
    matrix at once. Instead, it performs a binary search within each row individually. Therefore, the time complexity is linear with respect
    to the number of rows and logarithmic with respect to the number of columns.

    The space complexity of the searchMatrix function is O(1), as it only uses a constant amount of extra space to store the pointers.
    """

    # initialize the left and right pointers
    l = 0
    ROW_LENGTH = len(matrix[0])
    r = ROW_LENGTH - 1

    for row in matrix:
        # if the target is within the range of the current row, we perform a binary search on the row
        if row[0] <= target and target <= row[ROW_LENGTH - 1]:
            while l <= r:
                m = (l + r) // 2
                if row[m] == target:
                    return True
                elif row[m] < target:
                    l = m + 1
                elif row[m] > target:
                    r = m - 1
            # if the pointers meet and we still haven't found the target, it means that the target is not
            # in the list and so we return False
            return False
    # if the target is not within the range of the current row, we move on to the next row
    # if we have iterated through all the rows and still haven't found the target, it means that the target
    # is not in the matrix and so we return False
    return False


# ------------- 23. Koko Eating Bananas - Leetcode 875 - Medium ---------------
def minEatingSpeed(piles, h):
    """
    COMPLEXITY:

    The time complexity of the minEatingSpeed function is O(n log m), where n is the length of the piles list and m is the maximum number of bananas in a pile.
    This is because the function uses a BINARY SEARCH approach to find the minimum speed.
    The function performs a constant amount of work for each iteration, and the number of iterations is bounded by the length of the piles list.
    Additionally, the function performs a binary search on the possible speeds, which takes O(log m) time.

    The space complexity of the minEatingSpeed function is O(1), as it only uses a constant amount of extra space to store the pointers and the result variable.
    """

    # the minimum speed is 1 and the maximum speed is the maximum number of bananas in a pile
    l = 1
    r = max(piles)
    res = max(piles)

    # we perform a binary search on the possible speeds
    while l <= r:
        # calculate the middle speed
        k = (l + r) // 2
        hours = 0
        # calculate the number of hours it takes to eat all the bananas at the current speed (k)
        for p in piles:
            hours += math.ceil(p / k)
        # if the number of hours it takes to eat all the bananas at the current speed is less than
        # or equal to h, we update the result to be the minimum of the current result and the current
        # speed we also move the right pointer to the left so as to decrease the speed since we want
        # to find the minimum speed
        if hours <= h:
            res = min(res, k)
            r = k - 1
        # if the number of hours it takes to eat all the bananas at the current speed is greater than h,
        # it means our current speed is too small so we move the left pointer to the right so as to
        # increase the speed
        else:
            l = k + 1
    return res

# --------- 24. Find Minimum in Rotated Sorted Array - Leetcode 153 - Medium ------------


def findMin(nums):
    """
    COMPLEXITY:

    The time complexity of the findMin function is O(log n), where n is the length of the nums list.
    This is because the function uses a BINARY SEARCH approach to find the minimum value.

    The space complexity of the findMin function is O(1), as it only uses a constant amount of extra
    space to store the pointers and the result variable.
    """

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
        # If the middle element is greater than or equal to the element at the left pointer,
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

# ---------- 25. Search in Rotated Sorted Array - Leetcode 33 - Medium -------------


def search(nums, target):
    """
    COMPLEXITY:

    The time complexity of the search function is O(log n), where n is the length of the nums list.
    This is because the function uses a BINARY SEARCH approach to find the target value.

    The space complexity of the search function is O(1), as it only uses a constant amount of extra
    space to store the pointers.
    """

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
            # if the target is less than the number at the middle OR if the target is
            # greater than the number at the right pointer, there is no point in looking
            # at the right sorted portion, so we update our pointers to concentrate our
            # search on the left sorted portion
            if target < nums[mid] or target > nums[r]:
                r = mid - 1
            # otherwise, our target is surely in the right sorted portion and we change
            # our pointers to concentrate on this region
            else:
                l = mid + 1
    # when the target is not  in our list of numbers, we just return -1
    return -1


# ---------- 26. Time Based Key-Value Store - Leetcode 981 - Medium -------------
class TimeMap:

    def __init__(self):
        # initialize a dictionary to store the key-value pairs
        # key = string, value = [list of [value, timestamp] pairs]
        self.store = {}

    def set(self, key, value, timestamp):
        # if the key is already in the dictionary, we append the value and timestamp to the list
        if key in self.store:
            self.store[key].append([value, timestamp])
        # otherwise, we create a new list with the value and timestamp and add it to the dictionary
        else:
            self.store[key] = [[value, timestamp]]

    def get(self, key, timestamp):
        # initialize a variable to store the result
        res = ""
        # if the key is not in the dictionary, we return an empty string
        if key in self.store:
            # we get the list of [value, timestamp] pairs for the key
            values = self.store[key]
        else:
            values = []

        # binary search
        l = 0
        r = len(values) - 1
        while l <= r:
            m = (l + r) // 2
            # when the timestamp at the middle index is less than or equal to the input timestamp,
            # it means that the timestamp at the middle index is the closest to the input timestamp
            # and so we update our result to be the value at the middle index
            # we also move the left pointer to the right so as to look for a closer timestamp
            if values[m][1] <= timestamp:
                res = values[m][0]
                l = m + 1
            # when the input timestamp is less than the timestamp at the middle index, it means that
            # the timestamp at the middle index is too far from the input timestamp and so we move
            else:
                r = m - 1
        return res


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### SLIDING WINDOW ####
# ---------- 27. Best Time to Buy and Sell Stock - Leetcode 121 - Easy -------------
def maxProfit(prices):
    """
    COMPLEXITY:

    The time complexity of the maxProfit function is O(n), where n is the length of the prices list.
    This is because the function uses a SLIDING WINDOW approach to iterate through the list once.

    The space complexity of the maxProfit function is O(1), as it only uses a constant amount of extra
    space to store the pointers and the result variable.    
    """

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


# ---------- 28. Longest Substring Without Repeating Characters - Leetcode 3 - Medium -------------
def lengthOfLongestSubstring(s):
    """
    COMPLEXITY:

    The time complexity of the lengthOfLongestSubstring function is O(n), where n is the length of the input string s.
    This is because the function uses a SLIDING WINDOW approach to iterate through the string once.

    The space complexity of the lengthOfLongestSubstring function is O(n), where n is the length of the input string s.
    This is because the function uses a set to store the characters in the sliding window, and the maximum size of the set
    is proportional to the length of the input string.
    """

    # create a set that'll store all unique non-repeating characters in our sliding window
    charSet = set()
    # initialize left pointer
    l = 0
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

# --------- 29. Longest Repeating Character Replacement - Leetcode 424 - Medium ------------


def characterReplacement(s, k):
    """
    COMPLEXITY:

    The time complexity of the characterReplacement function is O(n), where n is the length of the input string s.
    This is because the function uses a SLIDING WINDOW approach to iterate through the string once.

    The space complexity of the characterReplacement function is O(1), as it only uses a constant amount of extra
    space to store the pointers and the result variable.
    """

    # initialize a dictionary that'll store the running occurrences
    count = {}
    res = 0
    # initialize left pointer
    l = 0

    # loop through the input string with the iterator acting as the right pointer
    for r in range(len(s)):
        # increment the count of the character at the right pointer in your dictionary
        char = s[r]
        if char in count:
            count[char] += 1
        else:
            count[char] = 1
        # check whether a character replacement can even be made
        # the logic of this is that if the length of the current substring window minus the
        # maximum count of a character in the current substring is greater than k, then
        # we know that we can't make a character replacement
        # it makes sense to replace the characters that are NOT the most frequent character
        # in the current substring because we can replace them with the most frequent character
        # replacements (k) in the current substring and still have a valid substring
        if (r - l + 1) - max(count.values()) > k:
            # when this occurs, we need to decrement the count of the character at the left
            # pointer in the dictionary since we're going to move the left pointer rightwards
            # to make a valid substring that can be made with k character replacements
            count[s[l]] -= 1
            l += 1
        # the maximum length of the substring with repeating characters will be the larger of
        # the previous window length and the current sliding window length
        res = max(res, r - l + 1)
    return res
