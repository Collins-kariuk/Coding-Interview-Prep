#### ARRAYS AND HASHING #####

# --------- 1. Contains Duplicate - Leetcode 217 - Easy ------------
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

# --------- 16. Evaluate Reverse Polish Situation - Leetcode 150 - Medium ------------


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
