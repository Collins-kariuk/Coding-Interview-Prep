import math
import collections
from collections import defaultdict, deque
import heapq

#### ARRAYS AND HASHING #####

# --------- 1. Contains Duplicate - Leetcode 217 - Easy ------------


def containsDuplicate(nums):
    """
    COMPLEXITY:

    The time complexity of the containsDuplicate function is O(n), where n is the length of the
    input list nums. This is because the function iterates over each element in nums once,
    performing constant-time operations such as set membership checks and set additions.

    The space complexity of the containsDuplicate function is O(n), where n is the length of the
    input list nums. This is because the function uses a set, hashSet, to store the unique
    elements of nums. The size of the set depends on the number of unique elements in nums, which
    can be at most n.
    """

    # initialize a set to store the unique elements of nums
    hashSet = set()
    # iterate through the list
    for n in nums:
        # if the element is already in the set, we return True
        if n in hashSet:
            return True
    # if the element is not in the set, we add it to the set
    hashSet.add(n)
    # if we get to the end of the list, it means that there are no duplicates
    return False

# --------- 2. Valid Anagram - Leetcode 242 - Easy ------------


def convertToDict(someString):
    stringDict = {}
    for c in someString:
        if c in stringDict:
            stringDict[c] += 1
        else:
            stringDict[c] = 1
    return stringDict


def isAnagram(s, t):
    """
    COMPLEXITY:

    The space complexity of the isAnagram function is O(n), where n is the length of the input
    strings s and t. This is because the function uses dictionaries sDict and tDict to store
    the frequency of each character in the strings. The size of the dictionaries depends on the
    number of unique characters in the strings, which can be at most n.

    The time complexity of the convertToDict function is O(n), where n is the length of the input
    string. This is because the function iterates over each character in the string once,
    performing constant-time operations such as dictionary lookups and updates.

    Therefore, the overall time complexity of the isAnagram function is O(n), where n is the length
    of the input strings s and t. This is because the function calls convertToDict twice, which
    has a time complexity of O(n), and then compares the two dictionaries, WHICH ALSO takes O(n)
    time.
    """

    # convert the strings to dictionaries that store the frequency of each character
    sDict = convertToDict(s)
    tDict = convertToDict(t)

    # s and t are anagrams only if the counts of the characters in both strings are the same
    return sDict == tDict

# --------- 3. Two Sum - Leetcode 1 - Easy ------------


def twoSum(nums, target):
    """
    COMPLEXITY:

    The space complexity of the twoSum function is O(n), where n is the length of the input list
    nums. This is because the function uses a dictionary, twoSumDict, to store the elements of nums
    as keys and their corresponding indices as values. The size of the dictionary depends on the
    number of unique elements in nums, which can be at most n.

    The time complexity of the twoSum function is O(n), where n is the length of the input list
    nums. This is because the function iterates over each element in nums once, performing
    constant-time operations such as dictionary lookups and updates. The worst-case scenario occurs
    when the target sum is achieved with the last two elements of nums, resulting in a linear time
    complexity.
    """

    twoSumDict = {}  # key:value = num:index

    for i in range(len(nums)):
        # calculate the second number that we need to achieve the target sum
        num2 = target - nums[i]
        # if the second number is already in the dictionary, we return the indices
        if num2 in twoSumDict:
            return [i, twoSumDict[num2]]
        # if second number is not in the dictionary, we add the current number to the dictionary
        else:
            twoSumDict[nums[i]] = i

# --------- 4. Group Anagrams - Leetcode 49 - Medium ------------


def groupAnagrams(strs):
    """
    COMPLEXITY:

    The space complexity of the groupAnagrams function is O(n), where n is the length of the input
    list strs. This is because the function uses a DICTIONARY, anagramDict, to store the anagrams
    as keys and their corresponding indices as values. The size of the dictionary depends on the
    number of UNIQUE ANAGRAMS in strs, which can be at most n.

    The time complexity of the sorted function is O(n log n), where n is the length of the input
    string s. This is because the sorted function uses a SORTING algorithm that has a time
    complexity of O(n log n) in the AVERAGE CASE. Therefore, the OVERALL time complexity of the
    groupAnagrams function is O(n * m log m), where n is the LENGTH OF THE INPUT LIST strs and m is
    the MAXIMUM LENGTH OF A STRING in strs. This is because the function iterates over each element
    in strs and performs the sorted function on each string, which has a time complexity of
    O(m log m).
    """

    anagramDict = {}  # key:value = sortedString:[list of anagrams]

    for s in strs:
        # The sorted function returns a sorted list of the characters in a string and the join
        # function converts the list back into a string
        sortedS = "".join(sorted(s))

        # If the sorted string is already in the anagramDict, we append the current string to the
        # list of anagrams. Otherwise, we create a new key-value pair in the anagramDict
        if sortedS in anagramDict:
            anagramDict[sortedS].append(s)
        else:
            anagramDict[sortedS] = [s]

    # returns a list of all the values in anagramDict.
    return list(anagramDict.values())


def groupAnagrams2(strs):
    """
    COMPLEXITY:

    (I think the space complexity is O(n) where n is the length of the input list strs because we
    use a dictionary to store the anagrams as keys and their corresponding indices as values. The
    size of the dictionary depends on the number of unique anagrams in strs, which can be at most
    n.)

    The time complexity of the function is O(NK), where N is the length of the input list strs and
    K is the maximum length of a string in strs. This is because we iterate through each string in
    strs and for each string, we iterate through each character to update the count list. The
    overall time complexity is determined by the NUMBER of strings and the MAXIMUM length of a
    string

    It's worth noting that the defaultdict lookup and append operations have an average time
    complexity of O(1), so they do not significantly affect the overall time complexity of the
    function
    """

    # defaultdict is a subclass of dict that provides a default value for a key that does not exist
    ans = collections.defaultdict(list)
    for s in strs:
        # we want to count the number of occurrences of each character in the string (26 represents
        # the number of letters in the alphabet)
        count = [0] * 26
        for c in s:
            # ord() returns the ASCII value of the character
            # for example, ord("a") returns 97 and ord("b") returns 98, so ord("b") - ord("a") = 1
            # which is the index of the count list for the character b
            count[ord(c) - ord("a")] += 1
        # we convert the count list to a tuple because lists are not hashable and cannot be used as
        # dictionary keys and tuples are hashable
        ans[tuple(count)].append(s)
    # returns a list of all the values in ans
    return ans.values()

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
    """

    numDict = {}
    for num in nums:
        if num in numDict:
            numDict[num] += 1
        else:
            numDict[num] = 1

    # a modified version of bucket sort
    frequencies = [[] for _ in range(len(nums) + 1)]

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

    The space complexity of the isValidSudoku function is O(1) because the space used by the cols,
    rows, and squares dictionaries is CONSTANT and does not depend on the size of the input.

    The time complexity of the isValidSudoku function is O(1) because the function iterates through
    a FIXED-SIZE 9x9 Sudoku board. The number of iterations is constant and does not depend on the
    size of the input. Therefore, the time complexity is constant.
    """

    # initialize the rows, cols, and squares dictionaries
    # the keys are the row/col/square numbers and the values are sets
    # the sets will contain the numbers that are in that particular row/col/square (will be
    # populated as we iterate through the board so we can check for duplicates in later iterations)
    # cols might look like {0: {5, 7}, 1: {1, 2, 3}, 2: {4, 6}}
    cols = collections.defaultdict(set)
    # rows might look like {0: {5, 7}, 1: {1, 2, 3}, 2: {4, 6}}
    rows = collections.defaultdict(set)
    # key = (r // 3, c // 3) so squares might look like {(0, 0): {5, 7}, (0, 1): {1, 2, 3},
    # (0, 2): {4, 6}}
    squares = collections.defaultdict(set)

    # iterate through the board
    for r in range(9):
        for c in range(9):
            # we don't care about the empty cells
            if board[r][c] == ".":
                continue
            # check if the number is already in the row/col/square
            # remember that the square is determined by the row and column number
            # for example, the square that contains the number in row 2 and column 3 is
            # (2 // 3, 3 // 3) = (0, 1)
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

    The space complexity of the isPalindrome function is O(n), where n is the length of the input
    string s. This is because the function CREATES 2 LISTS, original and reversedOriginal, to store
    the alphanumeric characters of the string. The size of these lists will be proportional to the
    length of the input string.

    The time complexity of the isPalindrome function is O(n), where n is the length of the input
    string s. This is because the function ITERATES through each character of the string once to
    populate the original list. Additionally, the function creates the reversedOriginal list by
    REVERSING the original list, which takes O(n) time. Finally, the function COMPARES the original
    and reversedOriginal lists, which also takes O(n) time.
    """

    # initialize two lists to store the original and reversed strings respectively
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

    The space complexity of the twoSumSorted function is O(1) because it uses a constant amount of
    extra space regardless of the input size. It only uses a few variables to store the pointers
    and the current sum.

    The time complexity of the twoSumSorted function is O(n), where n is the length of the numbers
    list. This is because the function uses a TWO-POINTER APPROACH to iterate through the list. The
    pointers move towards each other until they meet, and at each iteration, the function compares
    the current sum with the target value. Since the list is sorted, the function can determine
    whether to move the left pointer or the right pointer based on the comparison. 
    """

    # initialize two pointers, one at the start and one at the end
    l = 0
    r = len(numbers) - 1

    # iterate until the pointers meet
    while l < r:
        # calculate the current sum
        currSum = numbers[l] + numbers[r]
        # if the current sum is greater than the target, we need to decrease the sum so we move the
        # right pointer to the left
        if currSum > target:
            r -= 1
        elif currSum < target:
            l += 1
        # the current sum is equal to the target, we return the indices (1-indexed)
        else:
            return [l + 1, r + 1]

# --------- 12. 3Sum - Leetcode 15 - Medium ------------


def threeSum(nums):
    """
    COMPLEXITY:

    The space complexity of the threeSum function is O(n), where n is the length of the nums list.
    This is because the function uses additional space to store the result array res, which can
    potentially contain all possible unique triplets that sum up to 0. In the worst case scenario,
    the size of res can be O(n^2) if all elements in nums form unique triplets.
    nums = [-1, -2, -3, ..., -n, 1, 2, 3, ..., n]

    The time complexity of the threeSum function is O(n^2). The function first SORTS the nums list,
    which takes O(n log n) time. Then, it iterates through each element in the sorted list,
    resulting in O(n) iterations. Within each iteration, the function uses a two-pointer approach
    to find the remaining two numbers that sum up to the negation of the current number. This
    TWO-POINTER APPROACH takes O(n) time in the WORST CASE scenario, as the pointers can
    potentially traverse the entire list. Therefore, the overall time complexity is
    O(n log n + n^2), which SIMPLIFIES to O(n^2).
    """

    nums.sort()
    res = []

    for i in range(len(nums)):
        # when the number is greater than 0, we can safely break out of the loop because we know
        # that the numbers that follow (after sorting) will be GREATER THAN 0 and so the sum of 3
        # numbers will never be 0
        if nums[i] > 0:
            break
        # when the number is equal to the number before it, we can safely SKIP IT because we've
        # already considered it in a previous iteration
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # initialize the left and right pointers
        l = i + 1
        r = len(nums) - 1

        while l < r:
            # calculate the sum of the 3 numbers
            currSum = nums[i] + nums[l] + nums[r]
            # when the sum is less than 0, we increment the left pointer by one so as to increase
            # the sum
            if currSum < 0:
                l += 1
            # when the sum is greater than 0, we decrement the right pointer by one so as to
            # decrease the sum
            elif currSum > 0:
                r -= 1
            # when the sum is equal to 0, we append the 3 numbers to the result array
            else:
                res.append([nums[i], nums[l], nums[r]])
                # we change the states of the pointers by one so as to look for other combinations
                # that sum up to 0
                l += 1
                r -= 1
                # when the number at the left pointer is equal to the number before it, we
                # increment the left pointer by one so as to skip it if we don't check whether the
                # number at the left pointer is equal to the number before it, we'll end up with
                # duplicate doublets for our mini 2Sum problem which we do not want since the
                # question asks for unique triplets
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                # when the number at the right pointer is equal to the number before it, we
                # decrement the right pointer by one so as to skip it
                while l < r and nums[r] == nums[r + 1]:
                    r -= 1
    # return the result array
    return res

# --------- 13. Container with Most Water - Leetcode 11 - Medium ------------


def maxArea(height):
    """
    COMPLEXITY:

    The time complexity of the maxArea function is O(n), where n is the length of the height list.
    This is because the function uses a TWO-POINTER approach to iterate through the list once.

    The space complexity of the maxArea function is O(1), as it only uses a constant amount of
    extra space to store the pointers and the result variable.
    """

    # initialize pointers
    l = 0
    r = len(height) - 1
    res = 0

    while l < r:
        # calculate the current area at the specific point in the iteration it is basic equation of
        # base*height where the base is the difference in the pointers and the height is the
        # smaller of the 2 values at the left and right pointers
        currArea = (r - l) * min(height[l], height[r])
        # the current maximum volume at the specific point in the iteration is just the bigger of
        # the previous volume and the current volume
        res = max(res, currArea)
        # when the height at the left pointer is smaller than the height at the right pointer we
        # increment the left pointer by one so as to still preserve the bigger height at the right
        # pointer since that height may be the smaller of 2 heights later in the iteration
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    # return the maximum volume
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

    The space complexity of the isValid function is O(n), where n is the length of the input string
    s. This is because the function uses a stack to store opening brackets, and the maximum size of
    the stack is proportional to the number of opening brackets in the string.

    The time complexity of the isValid function is O(n), where n is the length of the input string
    s. This is because the function iterates through each character in the string once, performing
    constant-time operations for each character.
    """

    # initialize a stack
    stack = []
    # a dictionary that maps the closing bracket to the opening bracket
    # key:value = closing bracket:opening bracket
    closeToOpen = {"]": "[", "}": "{", ")": "("}

    for c in s:
        # if the character is a closing bracket
        if c in closeToOpen:
            # if the stack is not empty and the character at the top of the stack is the opening
            # bracket of the current closing bracket, we pop the opening bracket from the stack
            # else, we return False
            if len(stack) != 0 and stack[-1] == closeToOpen[c]:
                stack.pop()
            # when the stack is empty or if the open parens character at the top of the stack does
            # not match the open parens counterpart of the closing  parens we're looking at, then
            # it means that the input string is not a valid parens

            # in the case of the stack being empty, a sample input string would be '(()))[]{}'
            # whereby the time we get to the third closing ) parens, the stack will be empty since
            # 2 pops of ( will have been made in prior iterations

            # in the case of the open parens character at the top of the stack not matching the
            # open parens counterpart of the closing parens we're looking at, a sample string would
            # be '[{]}' whereby the stack will be non-empty but by the time we get to the third
            # character, the closing parens ], the character at the top of the stack will be the
            # prior { which does not match the open parens counterpart of ]
            else:
                return False
        # if the character is an opening bracket, we push it on top of the stack
        else:
            stack.append(c)
    # if the stack is empty, it means that all the opening brackets have been popped and there are
    # no closing brackets left and so the input string is valid
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

    The space complexity of the evalRPN function is O(n), where n is the number of tokens in the
    input list. This is because the function uses a stack to store the operands, and the size of
    the stack GROWS LINEARLY with the number of tokens.

    The time complexity of the evalRPN function is also O(n), where n is the number of tokens in
    the input list. This is because the function iterates through each token once and performs
    constant-time operations for each token. Therefore, the time complexity is linear with respect
    to the NUMBER OF TOKENS.
    """

    # initialize a stack to store the operands
    stack = []

    for token in tokens:
        # push the operands onto the stack
        if token not in "+-*/":
            stack.append(int(token))
        else:
            # pop the operands in the correct order
            num2 = stack.pop()
            num1 = stack.pop()

            # perform the operation and push the result back onto the stack
            if token == "+":
                stack.append(num1 + num2)
            elif token == "-":
                stack.append(num1 - num2)
            elif token == "*":
                stack.append(num1 * num2)
            elif token == "/":
                # use integer division that truncates towards zero
                stack.append(int(num1 / num2))

    # the result will be at the top of the stack
    return stack[0]

# --------- 18. Generate Parentheses - Leetcode 22 - Medium ------------


def generateParenthesis(n):
    """
    COMPLEXITY:

    The space complexity of the generateParenthesis function is O(n), where n is the input
    parameter representing the number of pairs of parentheses. This is because the function uses a
    stack to store the parentheses combinations, and the maximum size of the stack at any given
    time is n.
    The stack indeed stores one potential parentheses combination at a time, but the maximum length
    of this combination can be up to 2n (for n pairs of parentheses, each pair consists of an
    opening and a closing bracket). Therefore, the space complexity is O(n) because the maximum
    size of the stack (or the maximum length of a single combination) is proportional to the input
    size n.
    Additionally, the recursive nature of the function also contributes to the space complexity.
    Each recursive call to backtrack adds a new level to the call stack. In the worst-case
    scenario, the depth of recursion (i.e., the maximum height of the implicit call stack) can be
    up to 2n, which also contributes to the O(n) space complexity.
    So, in summary, the space complexity of the function is O(n) due to the MAXIMUM SIZE OF THE
    STACK used to store a single parentheses combination and the MAXIMUM DEPTH OF THE RECURSIVE
    CALL STACK.

    The time complexity of the generateParenthesis function is O(4^n/n^(1/2)), which can be
    approximated as O(4^n). This is because the function uses a backtracking approach to generate
    all the valid parentheses combinations. In the worst case, the function explores all possible
    combinations, which is exponential with respect to the input size. The factor of 4 comes from
    the two choices for each parentheses position (open or closed), and the division by n^(1/2)
    accounts for the number of invalid combinations that are pruned during the backtracking process
    """

    # initialize a stack to store the parentheses intermediately
    # at any given time in the backtracking process, the stack will store only one valid
    # parentheses combination
    stack = []
    # initialize the result list to store all the valid parentheses combinations
    res = []

    def backtrack(openCount, closedCount):
        """
        The purpose of the backtrack function is to generate all the valid parentheses
        combinations. The function uses a backtracking approach to generate all the combinations.
        The function takes in two parameters, openCount and closedCount, which represent the number
        of open and closed parentheses respectively.

        Yes, that is correct. At any given time during the backtracking process, the stack will
        only be storing one potential parentheses combination. This is because the function uses a
        backtracking approach, where it explores all possible combinations by adding and removing
        parentheses from the stack. Each time a valid combination is found, it is appended to the
        result list, and then the function continues exploring other possibilities.
        """

        # the base case is when the number of open and closed parentheses is equal to n which means
        # that we have generated a valid parentheses combination
        if openCount == closedCount == n:
            # we append the parentheses combination to the result list and return
            res.append("".join(stack))
            return

        # as long as the number of open parentheses is less than n, we can add an open parentheses
        # to the stack
        if openCount < n:
            stack.append("(")
            # we increment the openCount by one since we have added an open parentheses
            # we keep the closedCount the same since we have not added a closed parentheses
            # this is done so that future recursive calls can keep track of the number of open and
            # closed parentheses
            backtrack(openCount + 1, closedCount)
            # stack is a global variable and so we need to pop the open parentheses that we added
            # so that we can explore other possibilities
            stack.pop()

        if closedCount < openCount:
            stack.append(")")
            backtrack(openCount, closedCount + 1)
            stack.pop()

    # we call the backtrack function with the initial openCount and closedCount set to 0
    backtrack(0, 0)
    return res

# ------------- 19. Daily Temperatures - Leetcode 739 - Medium ---------------


def dailyTemperatures(temperatures):
    """
    A monotonic stack is a stack that either strictly increases or strictly decreases. In this
    problem, we use a monotonic decreasing stack to keep track of the indices of the temperatures
    that we have seen so far. The stack will store the indices in decreasing order of temperature.

    COMPLEXITY:

    The time complexity of the dailyTemperatures function is O(n), where n is the length of the
    temperatures list. This is because we iterate through each temperature once and perform
    CONSTANT-TIME OPERATIONS for each temperature.

    The space complexity of the function is O(n) as well. This is because we use a stack to store
    the indices of the temperatures that we have seen so far alongside the temperature. The size of
    the stack grows linearly with the number of temperatures, so the space complexity is also
    linear with respect to the length of the temperatures list.
    """

    # initialize the result array
    # we initialize it to all zeros because the default value when we do not find a warmer day is 0
    res = [0] * len(temperatures)
    # initialize a stack to store the indices of the temperatures that we have seen so far
    # alongside the temperature
    stack = []  # pair: [temp, index]

    for currIndex, temperature in enumerate(temperatures):
        # if the stack is not empty and the current temperature is greater than the temperature at
        # the top of the stack we pop the temperature and index from the stack and calculate the
        # number of days between the current index and the index at the top of the stack we then
        # update the result array at the index at the top of the stack with the number of days we
        # continue popping from the stack until the stack is empty or the current temperature is
        # less than the temperature at the top of the stack
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

    The time complexity of the minEatingSpeed function is O(n log m), where n is the length of the
    piles list and m is the maximum number of bananas in a pile. This is because the function uses
    a BINARY SEARCH approach to find the minimum speed. The function performs a constant amount of
    work for each iteration, and the number of iterations is bounded by the length of the piles
    list. Additionally, the function performs a binary search on the possible speeds, which takes
    O(log m) time.

    The space complexity of the minEatingSpeed function is O(1), as it only uses a constant amount
    of extra space to store the pointers and the result variable.
    """

    # the minimum speed is 1 and the maximum speed is the maximum number of bananas in a pile
    l = 1
    r = max(piles)
    # initialize the result to be the maximum number of bananas in a pile since we know that
    # regardless, the speed will have to be greater than or equal to the maximum number of bananas
    # in a pile
    res = max(piles)

    # perform a binary search on the possible speeds
    while l <= r:
        # calculate the middle speed
        k = (l + r) // 2
        hours = 0
        # calculate the number of hours it takes to eat all the bananas at the current speed (k)
        # we use the math.ceil function to round up the number of hours to the nearest whole number
        for p in piles:
            hours += math.ceil(p / k)
        # if the number of hours it takes to eat all the bananas at the current speed is less than
        # or equal to h, we update the result to be the minimum of the current result and the
        # current speed we also move the right pointer to the left so as to decrease the speed
        # since we want to find the minimum speed
        if hours <= h:
            res = min(res, k)
            r = k - 1
        # if the number of hours it takes to eat all the bananas at the current speed is greater
        # than h, it means our current speed is too small so we move the left pointer to the right
        # so as to increase the speed
        else:
            l = k + 1
    return res

# --------- 24. Find Minimum in Rotated Sorted Array - Leetcode 153 - Medium ------------


def findMin(nums):
    """
    COMPLEXITY:

    The time complexity of the findMin function is O(log n), where n is the length of the nums
    list. This is because the function uses a BINARY SEARCH approach to find the minimum value.

    The space complexity of the findMin function is O(1), as it only uses a constant amount of
    extra space to store the pointers and the result variable.
    """

    # variable that'll store the current minimum
    res = nums[0]
    # initialize pointers
    l = 0
    r = len(nums) - 1

    while l <= r:
        # if the number at the left pointer is less than the one at the right pointer, it means that
        # nums is already sorted and we can safely return the number at the left pointer or the
        # current minimum, whichever is smaller
        if nums[l] < nums[r]:
            res = min(res, nums[l])
            break

        # calculation of the location of the middle pointer
        mid = (l + r) // 2
        # before further comparison, the number at the middle pointer will serve as the minimum
        res = min(res, nums[mid])
        # If the middle element is greater than or equal to the element at the left pointer, it
        # indicates that the left segment of the sublist is already sorted. Due to the  array's
        # rotation, searching in the left segment is not logical, as it will always contain larger
        # values compared to the right segment. Therefore, our search should concentrate on the
        # right segment of the array.
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

    while l <= r:
        # calculate the middle pointer
        mid = (l + r) // 2
        # directly return if the target is equal to the number at the middle pointer
        if target == nums[mid]:
            return mid

        # left sorted portion
        # if the number at the middle is greater than the number at the left pointer, we are at the
        # left sorted portion
        if nums[l] <= nums[mid]:
            # if the target is greater than the number at the middle OR if the target is less than
            # the number at the left pointer, there is no point in looking at the left sorted
            # portion, so we update our pointers to concentrate our search on the right sorted
            # portion
            if target > nums[mid] or target < nums[l]:
                l = mid + 1
            # otherwise, our target is surely in the left sorted portion and we change
            # our pointers to concentrate on this region
            else:
                r = mid - 1

        # right sorted portion
        else:
            # if the target is less than the number at the middle OR if the target is greater than
            # the number at the right pointer, there is no point in looking at the right sorted
            # portion, so we update our pointers to concentrate our search on the left sorted
            # portion
            if target < nums[mid] or target > nums[r]:
                r = mid - 1
            # otherwise, our target is surely in the right sorted portion and we change our
            # pointers to concentrate on this region
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

    The time complexity of the lengthOfLongestSubstring function is O(n), where n is the length of
    the input string s. This is because the function uses a SLIDING WINDOW approach to iterate
    through the string once.

    The space complexity of the lengthOfLongestSubstring function is O(n), where n is the length of
    the input string s. This is because the function uses a set to store the characters in the
    sliding window, and the maximum size of the set is proportional to the length of the input
    string.
    """

    # create a set that'll store all unique non-repeating characters in our sliding window
    charSet = set()
    # initialize left pointer
    l = 0
    res = 0

    for r in range(len(s)):
        # if the character at the right pointer is already in the set, it means that we've
        # encountered a repeating character and we need to remove the character at the left pointer
        # from the set and advance the left pointer rightwards
        while s[r] in charSet:
            # 1. remove the character at the left pointer from the set
            # this is done because we need to keep track of the characters in the substring we're
            # currently looking at
            charSet.remove(s[l])
            # 2. advance the left pointer rightwards
            # this is done because we need to keep track of the characters in the substring we're
            # currently looking at we also need to advance the left pointer rightwards because we
            # need to remove the character at the left pointer from the set
            l += 1

        # add the character at the right pointer to the set, this is done regardless of whether the
        # character is repeated or not because we need to keep track of the characters in the
        # substring we're currently looking at
        charSet.add(s[r])
        # the maximum length of the substring without repeating characters will be the larger of
        # the previous length and the current length the current length is just the difference
        # between the right and left pointers plus one since we're dealing with 0-indexing
        res = max(res, r - l + 1)
    return res

# --------- 29. Longest Repeating Character Replacement - Leetcode 424 - Medium ------------


def characterReplacement(s, k):
    """
    COMPLEXITY:

    The time complexity of the characterReplacement function is O(n), where n is the length of the
    input string s. This is because the function uses a SLIDING WINDOW approach to iterate through
    the string once.

    The space complexity of the characterReplacement function is O(1), as it only uses a constant
    amount of extra space to store the pointers and the result variable.
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

        # check whether a character replacement can even be made the logic of this is that if the
        # length of the current substring window minus the maximum count of a character in the
        # current substring is greater than k, then we know that we can't make a character
        # replacement it makes sense to replace the characters that are NOT the most frequent
        # character in the current substring because we can replace them with the most frequent
        # character replacements (k) in the current substring and still have a valid substring

        # (r - l + 1) is the length of the current substring window
        # max(count.values()) is the maximum count of a character in the current substring
        if (r - l + 1) - max(count.values()) > k:
            # when this occurs, we need to decrement the count of the character at the left pointer
            # in the dictionary since we're going to move the left pointer rightwards to make a
            # valid substring that can be made with k character replacements
            count[s[l]] -= 1
            l += 1
        # the maximum length of the substring with repeating characters will be the larger of the
        # previous window length and the current sliding window length
        res = max(res, r - l + 1)
    return res

# --------- 30. Permutation in String - Leetcode 567 - Medium ------------


def dictify(s):
    someDict = {}
    for c in s:
        if c in someDict:
            someDict[c] += 1
        else:
            someDict[c] = 1
    return someDict


def checkInclusion(s1, s2):
    """
    COMPLEXITY:

    The time complexity of the checkInclusion function is O(n * m), where n is the length of s2 and
    m is the length of s1. This is because we iterate through s2 using the while loop, and for each
    iteration, we create a dictionary of the current window of length m using the dictify function.
    The dictify function has a time complexity of O(m) because it iterates through the characters
    of s1 and performs constant time operations to update the dictionary.

    The space complexity of the checkInclusion function is O(m), where m is the length of s1. This
    is because we create a dictionary s1Dict to store the character counts of s1, which can have at
    most m unique characters. Additionally, we create a curWindowDict dictionary for each window of
    length m in s2. Therefore, the space required is proportional to the length of s1.
    """

    # an edge case is when the length of s1 is greater than the length of s2
    if len(s1) > len(s2):
        return False

    # initialize left and right pointers
    l = 0
    r = len(s1) - 1
    # create a dictionary that'll store the character counts of s1
    s1Dict = dictify(s1)

    while r < len(s2):
        # create a dictionary that'll store the character counts of the current window
        curWindow = s2[l:r + 1]
        curWindowDict = dictify(curWindow)

        # if the character counts of the current window and s1 are the same, it means that the
        # current window is a permutation of s1
        if curWindowDict == s1Dict:
            return True
        # otherwise, we move the left and right pointers (the whole window) rightwards
        else:
            l += 1
            r += 1
    return False


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### LINKED LIST ####
# --------- 31. Reverse Linked List - Leetcode 206 - Easy ------------
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseList(head):
    """
    COMPLEXITY:

    The time complexity of the reverseList function is O(n), where n is the number of
    nodes in the linked list. This is because the function iterates through each node once.

    The space complexity of the reverseList function is O(1), which means it uses constant
    space. This is because the function only uses a constant amount of additional space to
    store the previous and current pointers, as well as the placeholder variable. The space
    used does not depend on the size of the input linked list.
    """

    # initialize the previous and current pointers
    prev = None
    curr = head

    # iterate through the linked list as long as the current pointer is not None
    while curr:
        # store the next node in a placeholder
        placeholder = curr.next
        # reverse the current node's next pointer to point to the previous node
        curr.next = prev
        # the next 2 steps set up the prev and curr pointers for the next iteration
        # move the previous pointer to the current node
        prev = curr
        # move the current pointer to the next node
        curr = placeholder
    # return the previous pointer since it'll be pointing to the last node of the original
    # linked list
    return prev


# --------- 32. Merge Two Sorted Lists - Leetcode 21 - Easy ------------
def mergeTwoLists(l1, l2):
    """
    COMPLEXITY:

    The time complexity of the mergeTwoLists function is O(n), where n is the total number
    of nodes of the larger of the two input linked lists. This is because the function
    iterates through each node once.

    The space complexity of the mergeTwoLists function is O(1), which means it uses constant
    space. This is because the function only uses a constant amount of additional space to
    store the temporary head (res), the tail pointer (tail), and the variables for iterating
    through the linked lists (l1 and l2). The space used does not depend on the size of the
    input linked lists.

    The new linked list that is being formed is not counted towards the space complexity
    because it's considered as the required output of the function, not additional space used
    by the function.
    """

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


# --------- 33. Reorder List - Leetcode 143 - Medium ------------
def reorderList(head):
    """
    Modify the head directly, without returning any value.
    The core strategy involves dividing the input linked list into two sections.
    To facilitate alternating between the segments, pointers to the heads of both halves are necessary.
    The main hurdle is that the second half of the list must be reversed for straightforward reintegration,
    as backtracking is not possible in a singly linked list.

    COMPLEXITY:

    The time complexity of the reorderList function is O(n), where n is the number of nodes in the linked list.
    This is because the function iterates through the linked list once to find the middle point, once to
    reverse the second half, and once to merge the two halves.

    The space complexity of the reorderList function is O(1), which means it uses constant space. This is
    because the function only uses a constant amount of additional space to store the pointers (slow, fast,
    second, prev, temp1, temp2) and variables for iterating through the linked list (first and second).
    The space used does not depend on the size of the input linked list.
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

    # we initialize a prev pointer to None because we need to reverse the second half
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


# --------- 34. Remove Nth Node from End of List - Leetcode 19 - Medium ------------
def removeNthFromEnd(head, n):
    """
    COMPLEXITY:

    The space complexity of the removeNthFromEnd function is O(1) because it uses a
    constant amount of extra space regardless of the size of the input.

    The time complexity of the function is O(n), where n is the length of the linked list.
    This is because the function iterates through the linked list twice: once to find the
    node to be removed (using the first pointer), and once to remove the node (using the
    second pointer).
    Both iterations take linear time proportional to the length of the linked list.
    """

    # Create a dummy node and attach it to the head of the input list.
    dummy = ListNode(val=0, next=head)

    # Initialize 2 pointers, first and second, to point to the dummy node.
    first = dummy
    second = dummy

    # Advances first pointer so that the gap between first and second is n nodes apart
    for _ in range(n + 1):
        first = first.next

    # While the first pointer does not equal null move both first and second to
    # maintain the gap and get nth node from the end
    while first != None:
        first = first.next
        second = second.next

    # Delete the node being pointed to by second.
    second.next = second.next.next

    # Return dummy.next
    return dummy.next

# --------- 35. Copy List with Random Pointer - Leetcode 138 - Medium ------------


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


def copyRandomList(head):
    """
    COMPLEXITY:

    The space complexity of the copyRandomList function is O(n), where n is the number of
    nodes in the linked list. This is because we create a dictionary oldToCopy that stores
    the old nodes as keys and the new nodes as values. The size of this dictionary is
    proportional to the number of nodes in the linked list.

    The time complexity of the function is also O(n), where n is the number of nodes in the
    linked list. This is because we iterate through the linked list twice: once to create a
    copy of the linked list without the random pointers or the next pointers, and once to set
    the next and random pointers of the copy nodes. Both iterations take linear time
    proportional to the number of nodes in the linked list.
    """

    # create a dictionary that'll store the old node as the key and the new node as the value
    oldToCopy = {None: None}  # key:value = oldNode:newNode

    # the first pass
    # create a copy of the linked list without the random pointers or the next pointers
    cur = head
    while cur:
        # create a new node with the same value as the current node
        copy = Node(x=cur.val, next=None, random=None)
        # add the current node and its copy to the dictionary
        oldToCopy[cur] = copy
        cur = cur.next

    # the second pass
    # set the next and random pointers of the copy nodes
    cur = head
    while cur:
        # get the copy of the current node
        copy = oldToCopy[cur]
        # set the next pointer of the copy node to be the copy of the next node which we can
        # get from the dictionary by accessing the next pointer of the current node
        copy.next = oldToCopy[cur.next]
        # set the random pointer of the copy node to be the copy of the random node which we
        # can get from the dictionary by accessing the random pointer of the current node
        copy.random = oldToCopy[cur.random]
        cur = cur.next

    # return the copy of the head node
    return oldToCopy[head]

# --------- 36. Add Two Numbers - Leetcode 2 - Medium ------------


def addTwoNumbers(l1, l2):
    """
    COMPLEXITY:

    The space complexity of the addTwoNumbers function is O(max(m, n)), where m and n are the
    lengths of the input linked lists l1 and l2 respectively. This is because we create a new 
    linked list to store the sum, which can have a maximum length of max(m, n) + 1.

    The time complexity of the function is O(max(m, n)), where m and n are the lengths of the
    input linked lists l1 and l2 respectively. This is because we iterate through the linked
    lists once, performing constant time operations for each node. The number of iterations
    is determined by the length of the longer linked list, which is max(m, n).
    """

    # initialize a dummy node that'll serve as a placeholder
    dummy = ListNode()
    # a pointer to the dummy node
    cur = dummy
    # initialize a carry variable that'll store the carry value
    carry = 0
    # we continue with the loop as long as both pointers to the input linked lists
    # are non-null and the carry value is not 0
    while l1 or l2 or carry != 0:
        # obtain the value of the current node in l1 and l2
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0

        # calculate the sum of the values of the current nodes in l1 and l2
        val = v1 + v2 + carry
        # since the value could be greater than 9, we need to calculate the carry value and
        # also the value of the current node in the resultant linked list
        carry = val // 10
        val = val % 10

        # create a new node with the calculated value
        cur.next = ListNode(val)

        # update pointers
        cur = cur.next
        # we only move the pointers to the input linked lists if they're not null since we
        # don't want to access the next pointer of a null node
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return dummy.next

# --------- 37. Linked List Cycle - Leetcode 141 - Easy ------------


def hasCycle(head):
    """
    COMPLEXITY:

    The space complexity of the hasCycle function is O(1) because it uses a constant
    amount of extra space to store the slow and fast pointers.

    The time complexity of the function is O(n), where n is the number of nodes in the
    linked list. This is because in the worst case scenario, the fast pointer will
    traverse the entire linked list before reaching the end or encountering a cycle.
    """

    # initialize the slow and fast pointers
    slow = head
    fast = head

    # we continue with the loop as long as the fast pointer is not null and the next
    # pointer of the fast pointer is not null
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        # if the slow and fast pointers are equal, it means that there is a cycle in the
        # linked list
        if slow == fast:
            return True
    # if the fast pointer reaches null, it means that there is no cycle in the linked list
    return False


# --------- 38. Find the Duplicate Number - Leetcode 287 - Medium ------------
def findDuplicate(nums):
    l = 0
    r = 1

    while r < len(nums):
        if nums[l] == nums[r]:
            return nums[l]
        r += 1

    l += 1
    r = l + 1

# --------- 39. LRU Cache - Leetcode 146 - Medium ------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### TREES ####
# --------- 40. Invert Binary Tree - Leetcode 226 - Easy ------------
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invertTree(root):
    """
    COMPLEXITY:

    The time complexity of the invertTree function is O(n), where
    n is the number of nodes in the binary tree. This is because
    the function visits each node once.

    The space complexity of the invertTree function is O(h), where
    h is the height of the binary tree. This is because the function
    uses the call stack to store recursive function calls, and the
    maximum depth of the call stack is equal to the height of the tree.
    """

    # base case
    if root == None:
        return None

    # swap the left and right children
    # temp = root.left
    # root.left = root.right
    # root.right = temp

    root.left, root.right = root.right, root.left

    # recursively call the method on the right and left children
    invertTree(root.left)
    invertTree(root.right)

    return root

# --------- 41. Maximum Depth of Binary Tree - Leetcode 104 - Easy ------------


def maxDepth(root):
    """
    COMPLEXITY:

    The time complexity of the maxDepth function is O(n), where n is the number of nodes
    in the binary tree. This is because the function visits each node once.

    The space complexity of the maxDepth function is O(h), where h is the height of the
    binary tree. This is because the function uses the call stack to store recursive
    function calls, and the maximum depth of the call stack is equal to the height of
    the tree.
    """

    # base case
    if root == None:
        return 0
    # recursively call the method on the right and left children
    # these recursive calls will return the maximum depth of the right and left subtrees
    leftDepth = maxDepth(root.left)
    rightDepth = maxDepth(root.right)
    # the maximum depth of the binary tree will be the larger of the maximum depth of the
    # right and left subtrees plus one (for the root node)
    return 1 + max(leftDepth, rightDepth)

# --------- 42. Diameter of Binary Tree - Leetcode 543 - Easy ------------


def diameterOfBinaryTree(root):
    """
    COMPLEXITY:

    The time complexity of the diameterOfBinaryTree function is O(n), where n is the number
    of nodes in the binary tree. This is because the function visits each node once during
    the depth-first search traversal.

    The space complexity of the diameterOfBinaryTree function is O(h), where h is the height
    of the binary tree. This is because the function uses the call stack to store recursive
    function calls, and the maximum depth of the call stack is equal to the height of the tree.
    """

    # initialize a variable that'll store the maximum diameter
    res = [0]

    def dfs(root):
        # base case
        # per convention, the height of a null node is -1 and the height of a leaf node is 0
        if root == None:
            return -1

        # recursively call the method on the right and left children to obtain their heights
        leftHeight = dfs(root.left)
        rightHeight = dfs(root.right)

        # the maximum diameter will be the larger of the previous maximum diameter and the
        # sum of the heights of the right and left subtrees plus 2 (for the root node)
        res[0] = max(res[0], 2 + leftHeight + rightHeight)

        # the height of the current node will be the larger of the heights of the right and
        # left subtrees plus 1 (for the current node)
        return 1 + max(leftHeight, rightHeight)

    # call the dfs function on the root node
    dfs(root)
    # return the maximum diameter
    return res[0]


# --------- 42. Balanced Binary Tree - Leetcode 110 - Easy ------------
def isBalanced(root):
    """
    COMPLEXITY:

    The time complexity of the isBalanced function is O(n), where n is the number of nodes in the
    binary tree. This is because the function visits each node once during the DFS traversal.

    The space complexity of the isBalanced function is O(h), where h is the height of the binary
    tree. This is because the function uses the call stack to store recursive function calls, and
    the maximum depth of the call stack is equal to the height of the tree.
    """

    def dfs(root):
        # base case
        # the first element is a boolean that INDICATES WHETHER THE TREE IS BALANCED and since, for
        # the base case, the node is null, the tree is balanced the second element is the HEIGHT OF
        # THE TREE
        if root == None:
            return [True, 0]

        # recursively call the method on the right and left children
        left = dfs(root.left)
        right = dfs(root.right)

        # the tree is balanced if the left and right subtrees are balanced and the difference
        # between their heights is less than or equal to 1
        balance = left[0] and right[0] and (abs(left[1] - right[1]) <= 1)

        # the height of the tree will be the larger of the heights of the right and left subtrees
        # plus 1 (for the current node)
        return [balance, 1 + max(left[1], right[1])]

    # call the dfs function on the root node and return the first element of the list because it
    # indicates whether the tree is balanced
    return dfs(root)[0]

# --------- 43. Same Tree - Leetcode 100 - Easy ------------


def isSameTree(p, q):
    """
    COMPLEXITY:

    The time complexity of the isSameTree function is O(n), where n is the number of nodes
    in the tree. This is because the function visits each node once during the recursive
    traversal.

    The space complexity of the isSameTree function is O(h), where h is the height of the
    tree. This is because the function uses the call stack to store recursive function
    calls, and the maximum depth of the call stack is equal to the height of the tree.
    """

    # base case
    # if both nodes are null, it means that the trees are the same
    if p == None and q == None:
        return True
    # if one of the nodes is null, it means that the trees are not the same
    # if the values of the nodes are not equal, it means that the trees are not the same
    if (p == None or q == None) or (p.val != q.val):
        return False

    # we recursively call the function on the left and right children of the nodes
    leftSide = isSameTree(p.left, q.left)
    rightSide = isSameTree(p.right, q.right)

    # the trees are the same if the left and right subtrees are the same
    return leftSide and rightSide

# --------- 44. Subtree of Another Tree - Leetcode 572 - Easy ------------


def isSubtree(root, subRoot):
    """
    Let's say:
    n is the number of nodes in the main tree (root) and m is the number of nodes
    in the subtree (subRoot).
    In the worst-case scenario, the function will have to compare the subRoot tree
    with every node's subtree in the main tree. The isSameTree function, which is used
    to check if two subtrees are identical, has a time complexity of O(min(n, m))
    because it needs to traverse every node in both trees.
    Thus, in the worst case, the isSubtree function is called once for each node in the
    main tree and for each call, isSameTree may potentially be called (in the worst case,
    if the trees are very unbalanced and the subtree is large compared to the main tree).
    This gives us a total time complexity of O(n * min(n, m)).

    The space complexity is determined by the depth of the recursion stack, which, in the
    worst case, can go as deep as the height of the trees. For a balanced tree, the height
    would be log(n) for the main tree and log(m) for the subtree. However, in the worst
    case for an unbalanced tree, the height could be as much as n for the main tree and m
    for the subtree.
    Therefore, the worst-case space complexity is O(n + m), corresponding to the depth of
    the recursion stack if both trees are completely unbalanced.
    """

    # base cases
    # technically, a null sub-root node is a subtree of any tree
    if subRoot == None:
        return True
    # technically, a null root node cannot have a subtree aside from a null subtree
    # (which we've checked above)
    if root == None:
        return False
    # if the values of the root and sub-root nodes are equal, we check whether the
    # trees are the same
    if isSameTree(root, subRoot):
        return True
    # recursively call the function on the left and right children of the nodes
    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)

# --------- 45. Lowest Common Ancestor of a Binary Search Tree - Leetcode 235 - Medium ------------


def lowestCommonAncestor(root, p, q):
    """
    COMPLEXITY:

    The time complexity of the lowestCommonAncestor function is O(log(n)) in the average case and
    O(n) in the worst case, where n is the number of nodes in the binary search tree. This is
    because the function traverses the tree in a binary search manner, comparing the values of p
    and q with the value of the current node to determine the next node to visit. In the average
    case, the function can eliminate half of the remaining nodes at each step, resulting in a
    logarithmic time complexity. However, in the worst case, where the tree is highly unbalanced,
    the function may need to visit all nodes, resulting in a linear time complexity.

    The space complexity of the function is O(1) in the average case and O(n) in the worst case.
    This is because the function uses a constant amount of additional space for the iterative
    approach, regardless of the size of the tree. However, in the worst case, where the tree is
    highly unbalanced and resembles a linked list, the function may need to recursively call
    itself for each node in the tree, resulting in a recursion stack that can go as deep as the
    height of the tree, which is O(n).
    """

    if p.val > root.val and q.val > root.val:
        # both p and q are greater than root, so lowest common ancestor must be in right subtree
        return lowestCommonAncestor(root.right, p, q)
    if p.val < root.val and q.val < root.val:
        # both p and q are less than root, so lowest common ancestor must be in left subtree
        return lowestCommonAncestor(root.left, p, q)
    # if neither of the above conditions are true, it means that p and q are on either side
    return root

# alternative (iterative) solution


def lowestCommonAncestor(root, p, q):
    while True:
        # if both p and q are greater than the current node, it means that the lowest common
        # ancestor must be in the right subtree
        if root.val < p.val and root.val < q.val:
            root = root.right
        # if both p and q are less than the current node, it means that the lowest common
        # ancestor must be in the left subtree
        elif root.val > p.val and root.val > q.val:
            root = root.left
        # if neither of the above conditions are true, it means that p and q are on either side
        # of the current node, so the current node is the lowest common ancestor
        else:
            return root

# --------------- 54. Validate Binary Search Tree - Leetcode 98 - Medium --------------


def isValidBST(root):
    """
    The time complexity of the isValidBST function is O(N), where N is the number of nodes in the
    binary tree. This is because the function performs a depth-first search traversal of the tree,
    visiting each node once.

    The space complexity of the function is O(N) as well. This is because the function uses a
    recursive approach to traverse the tree, and the maximum depth of the recursion is equal to the
    height of the tree, which can be N in the worst case for an unbalanced tree.
    """

    # the gist of the solution is that we need to check whether the current node's value is between
    # the minimum and maximum values we do this by recursively calling the function on the left and
    # right subtrees and updating the minimum and maximum values as we go along
    def dfs_helper(node, minVal, maxVal):
        # base case 1: if the node is null, we return True because an empty tree is a valid BST
        if not node:
            return True
        # base case 2: if the node's value is less than the minimum value or greater than the max
        # value, we return False
        if node.val <= minVal or node.val >= maxVal:
            return False

        # the recursive case: we call the function on the left and right subtrees
        # we update the minimum and maximum values as we go along

        # for the left subtree, the maximum value is the value of the current node and for the
        # right subtree, the minimum value is the value of the current node
        return dfs_helper(node.left, minVal, node.val) and dfs_helper(node.right, node.val, maxVal)

    # we set the initial minimum and maximum values to be the minimum and maximum values of a
    # 32-bit signed integer
    return dfs_helper(root, -2**31, 2**31 - 1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### BACKTRACKING ####
# --------- 46. Subsets - Leetcode 78 - Medium ------------


def subsets(nums):
    """
    COMPLEXITY:

    The time complexity of the subsets function is O(2^N), where N is the length of the input list
    nums. This is because for each element in nums, we have two choices: either include it in the
    subset or not include it. Therefore, the number of subsets grows exponentially with the size of
    the input list.

    The space complexity of the function is O(N), where N is the length of the input list nums.
    This is because we use a recursive approach to generate all subsets, and the maximum depth of
    the recursion  is equal to the length of the input list. Additionally, we use an auxiliary list
    subset to store the current subset, which can have a maximum size of N.
    """

    res = []
    # initialize a list that'll store the current subset
    subset = []

    def dfs(i):
        # base case
        # when i (the index of the current element) is greater than or equal to the length of the
        # input list, it means that we've reached the end of the list and we can add the current
        # subset to the result list the reason why we need to add a copy of the current subset is
        # because we're going to be modifying the subset in the recursive calls and we don't want
        # to add the modified subset to the result list
        if i >= len(nums):
            res.append(subset.copy())
            return

        # there are 2 decisions to make at each step:
        # 1. decision to include nums[i]
        subset.append(nums[i])
        dfs(i + 1)

        # 2. decision NOT to include nums[i]
        # we pop the last element from the subset list because we want to backtrack and try the
        # other decisions
        subset.pop()
        dfs(i + 1)

    dfs(0)
    return res

# --------- 47. Combination Sum - Leetcode 39 - Medium ------------


def combinationSum(candidates, target):
    """
    COMPLEXITY:

    The time complexity of the combinationSum function is exponential, specifically O(2^N),
    where N is the length of the candidates list. This is because for each element in
    candidates, we have two choices: either include it in the combination or not include it.
    Therefore, the number of combinations grows exponentially with the size of the input list.

    The space complexity of the function is O(N), where N is the length of the candidates list.
    This is because we use a recursive approach to generate all combinations, and the maximum
    depth of the recursion is equal to the length of the input list. Additionally, we use an
    auxiliary list cur to store the current combination, which can have a maximum size of N.
    """

    # initialize a list that'll store the current combination
    res = []

    def dfs(i, cur, total):
        """
        i: the index of the current element in candidates
        cur: the current combination
        total: the sum of the elements in the current combination
        """

        # base case number 1
        if total == target:
            # we add a copy of the current combination to the result list because we're
            # going to be modifying the current combination in the recursive calls and we
            # don't want to add the modified combination to the result list
            res.append(cur.copy())
            # we return because we don't want to continue with the recursive calls
            return

        # base case number 2
        # when i (the index of the current element) is greater than or equal to the length
        # of the input list, it means that we've reached the end of the list and we can
        # return because we don't want to continue with the recursive calls
        # we also return if the total is greater than the target because we don't want to
        # continue with the recursive calls since we know that the sum of the elements in
        # the current combination will be greater than the target
        if i >= len(candidates) or total > target:
            return

        # there are 2 decisions to make at each step:
        # 1. decision to include candidates[i]
        cur.append(candidates[i])
        # we pass i as the index of the current element because we can use the same element
        # multiple times
        dfs(i, cur, total + candidates[i])

        # 2. decision NOT to include candidates[i]
        cur.pop()
        # we pass i + 1 as the index of the current element because we can't use the same
        # element multiple times
        dfs(i + 1, cur, total)

    # we start the recursive calls at index 0 and with an empty combination and a total of 0
    dfs(i=0, cur=[], total=0)
    return res

# --------- 48. Word Search - Leetcode 79 - Medium ------------


def exist(board, word):
    """
    COMPLEXITY:

    The time complexity of the exist function is O(N * M * 4^L), where N is the number of rows in the
    board, M is the number of columns in the board, and L is the length of the word. This is because
    for each cell in the board, we perform a DFS in four directions (up, down, left, right) until we
    either find the word or reach the end of the board. The worst-case scenario is that we have to
    explore all possible paths, which gives us a time complexity of O(N * M * 4^L).

    The space complexity of the exist function is O(L), where L is the length of the word. This is because
    we use a set called path to keep track of the visited cells during the DFS. The maximum number of
    cells that can be visited at any given time is equal to the length of the word. Therefore, the space
    complexity is O(L).
    """

    # save the dimensions of the board
    ROWS = len(board)
    COLS = len(board[0])

    # initialize a set that'll store the visited cells
    path = set()

    def dfs(r, c, i):
        """
        r: the row of the current cell
        c: the column of the current cell
        i: the index of the current character in word
        """

        # base case 1
        # if i is equal to the length of word, it means that we've reached the end of the
        # word and we can return True
        if i == len(word):
            return True

        # base case 2
        # if r or c are out of bounds, it means that we've reached the end of the board or
        # we've reached a cell that we've already visited or the current character in the
        # board is not the same as the current character in word
        # in any of these cases, we return False
        if (r < 0 or c < 0 or r >= ROWS or c >= COLS) or (word[i] != board[r][c]) or ((r, c) in path):
            return False

        # we add the current cell to the set of visited cells
        path.add((r, c))

        # we recursively call the function on the cells to the right, left, top, and bottom
        # of the current cell
        res = (dfs(r + 1, c, i + 1) or
               dfs(r - 1, c, i + 1) or
               dfs(r, c + 1, i + 1) or
               dfs(r, c - 1, i + 1))
        # we remove the current cell from the set of visited cells because we're going to
        # be modifying the set in the recursive calls and we don't want to add the modified
        # set to the result list
        path.remove((r, c))
        return res

    # we iterate through each cell in the board and call the dfs function on each cell
    for r in range(ROWS):
        for c in range(COLS):
            if dfs(r, c, 0):
                return True

    # if we reach this point, it means that we haven't found the word in the board
    return False


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### GRAPHS ####
# --------- 49. Number of Islands - Leetcode 200 - Medium ------------
def numIslands(grid):
    """
    COMPLEXITY:

    The space complexity of the numIslands function is O(M * N), where M is the number of rows in
    the grid and N is the number of columns in the grid. This is because we use a set visited to
    keep track of the visited islands, which can store at most M * N island coordinates.

    The time complexity of the function is O(M * N), as we iterate through each individual grid
    cell once in the nested loops. Additionally, for each unvisited land cell, we perform a BFS
    operation, which can potentially visit all the cells in the grid in the worst case. Therefore,
    the overall time complexity is dominated by the BFS operation, resulting in O(M * N).
    """

    # edge case: when the grid is empty
    if len(grid) == 0:
        return 0

    # initialize the number of islands
    islands = 0

    # set that'll store the visited islands
    visited = set()

    # the dimensions of the grid
    ROWS = len(grid)
    COLS = len(grid[0])

    def bfs(r, c):
        """
        Conducting a breadth-first search to count the number of islands, while keeping track of the
        islands already visited. This ensures that we don't mistakenly revisit the same islands.

        r: the row of the current subgrid
        c: the column of the current subgrid 
        """

        # bfs is an iterative algorithm that needs a queue
        q = deque()
        # we add the island to the visited pile
        visited.add((r, c))
        # append the island we're at in the iteration in our bfs queue
        q.append((r, c))

        # traverse through the queue as long as it's non-empty thus "expanding our island"
        while q:
            # the subgrid coordinate at the top of our queue
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
                if (r in range(ROWS) and c in range(COLS)) and grid[r][c] == '1' and (r, c) not in visited:
                    # add to queue because we also have to run bfs on this cell as well
                    q.append((r, c))
                    # mark it as visited so that we don't visit it twice
                    visited.add((r, c))

    # looping through each individual grid
    for r in range(ROWS):
        for c in range(COLS):
            # if the subgrid is land and is not among the visited, do a BFS on it and increment the number
            # of islands
            if grid[r][c] == "1" and (r, c) not in visited:
                bfs(r, c)
                islands += 1

    # the final number of islands
    return islands

# --------- 50. Clone Graph - Leetcode 133 - Medium ------------


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def cloneGraph(node):
    """
    COMPLEXITY:

    The space complexity of the cloneGraph function is O(V), where V is the number of
    nodes in the graph. This is because we use a dictionary (oldToNew) to store the
    mapping between old nodes and new nodes, and the size of the dictionary will be
    proportional to the number of nodes in the graph.

    The time complexity of the cloneGraph function is O(V + E), where V is the number
    of nodes and E is the number of edges in the graph. This is because we perform a DFS
    traversal of the graph, visiting each node and its neighbors exactly once. The time
    complexity of the DFS traversal is O(V + E). Additionally, for each node, we create a
    copy of the node and its neighbors, which takes O(1) time. Therefore, the overall
    time complexity is O(V + E).
    """

    # a dictionary that'll store the old node as the key and the new node as the value
    oldToNew = {}  # key:value = oldNode:newNode

    def clone_dfs(node):
        """
        node: the current node in the DFS traversal
        """

        # base case
        # when the node is already in the dictionary, it means that we've already created
        # a copy of the node and we can return the copy
        if node in oldToNew:
            return oldToNew[node]

        # create a copy of the current node and add it to the dictionary
        copy = Node(node.val)
        oldToNew[node] = copy

        # recursively call the function on the neighbors of the current node
        for neighbor in node.neighbors:
            copy.neighbors.append(clone_dfs(neighbor))

        # return the copy of the current node
        return copy

    # call the clone_dfs function on the input node
    return clone_dfs(node) if node else None

# --------- 51. Pacific Atlantic Water Flow - Leetcode 417 - Medium ------------


def pacificAtlantic(heights):
    """
    COMPLEXITY:

    The space complexity of the pacificAtlantic function is O(R * C), where R is the number of rows
    in the grid and C is the number of columns in the grid. This is because we use two sets (pacific
    and atlantic) to store the visited cells, and in the worst case, all cells in the grid can be
    visited.

    The time complexity of the pacificAtlantic function is O(R * C), where R is the number of rows
    in the grid and C is the number of columns in the grid. This is because we perform a DFS traversal
    on the grid, visiting each cell exactly once. In the worst case, we may need to visit all cells
    in the grid.

    Additionally, the nested loops at the end of the function iterate through each cell in the grid,
    resulting in an additional time complexity of O(R * C).

    Therefore, the overall time complexity of the function is O(R * C), and the space complexity is
    O(R * C).
    """

    # save the dimensions of the grid
    ROWS = len(heights)
    COLS = len(heights[0])

    # the set of cells that can reach the pacific ocean and the atlantic ocean
    pacific = set()
    atlantic = set()

    def dfs(r, c, visit, prevHeight):
        """
        r: the row of the current cell
        c: the column of the current cell
        visit: the set of visited cells (either pacific or atlantic)
        prevHeight: the height of the previous cell
        """

        # base case
        # if the cell is already in the set of visited cells or if the cell is out of bounds
        # or if the height of the cell is less than the height of the previous cell (we are starting
        # from the ocean and working our way inland so we want the heights we meet inland to be
        # greater), we return
        if (r, c) in visit or r < 0 or c < 0 or r == ROWS or c == COLS or heights[r][c] < prevHeight:
            return

        # add the current cell to the set of visited cells
        visit.add((r, c))

        # recursively call the function on the cells to the right, left, top, and bottom
        dfs(r + 1, c, visit, heights[r][c])
        dfs(r - 1, c, visit, heights[r][c])
        dfs(r, c + 1, visit, heights[r][c])
        dfs(r, c - 1, visit, heights[r][c])

    # we start the recursive calls at the cells on the borders of the grid
    # the cells on the top border reach the pacific ocean and the cells on the bottom border
    # reach the atlantic ocean
    for col in range(COLS):
        dfs(0, col, pacific, heights[0][col])
        dfs(ROWS - 1, col, atlantic, heights[ROWS - 1][col])

    # the cells on the left border reach the pacific ocean and the cells on the right border
    # reach the atlantic ocean
    for row in range(ROWS):
        dfs(row, 0, pacific, heights[row][0])
        dfs(row, COLS - 1, atlantic, heights[row][COLS - 1])

    # we iterate through each cell in the grid and check if the cell can reach both oceans
    # and if it can, we add it to the result list
    res = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in pacific and (r, c) in atlantic:
                res.append([r, c])

    return res

# --------- 52. Course Schedule - Leetcode 207 - Medium ------------


def canFinish(numCourses, prerequisites):
    """
    COMPLEXITY:

    The space complexity of the canFinish function is O(N), where N is the number of courses. This
    is because we are using a dictionary (prerequisiteMap) to store the prerequisites for each
    course, and the size of the dictionary will be proportional to the number of courses.

    The time complexity of the canFinish function is O(N + E), where N is the number of courses and
    E is the number of prerequisites. This is because we need to populate the prerequisiteMap
    dictionary, which takes O(E) time as we iterate through the prerequisites. Then, we perform a
    DFS on each course, which takes O(N) time as we visit each course once. Therefore, the overall
    time complexity is O(N + E).
    """

    # initialize a dictionary that'll store the prerequisites for each course in an adjacency list
    # so each course is mapped to its prerequisites
    prerequisiteMap = {i: [] for i in range(numCourses)}

    # a more verbose way of writing the above line
    # prerequisiteMap = {}
    # for i in range(numCourses):
    #     prerequisiteMap[i] = []

    # populate the dictionary
    for course, prerequisite in prerequisites:
        prerequisiteMap[course].append(prerequisite)

    # visitSet = all courses along the current DFS path
    visitSet = set()

    def dfs(course):
        # base case 1 - if the course is in the visitSet, it means that we've encountered a cycle
        if course in visitSet:
            return False

        # base case 2 - if the course has no prerequisites, it means we can successfully complete
        # the course
        if prerequisiteMap[course] == []:
            return True

        # add the course to the visitSet
        visitSet.add(course)

        # recursively call the function on the prerequisites of the course
        for prerequisite in prerequisiteMap[course]:
            # immediately return False if we find a cycle (if we cannot complete a course)
            if not dfs(prerequisite):
                return False

        # remove the course from the visitSet because we're done with the current DFS path
        """
        The primary role of this line is to backtrack correctly. When the DFS for a particular
        course is complete (i.e., all courses dependent on this course have been visited and
        checked for cycles), we need to remove this course from the visitSet. This action signifies
        that we are backtracking from this course and it should not be considered part of the
        current DFS path anymore.
        """
        visitSet.remove(course)
        # remove the course from the prerequisiteMap because we've completed the course
        prerequisiteMap[course] = []
        return True

    # we call the dfs function on each course
    # we need to call the dfs function on each course because there may be multiple disconnected
    # components in the graph
    for course in range(numCourses):
        if not dfs(course):
            return False

    # if we reach this point, it means that we can complete all courses
    return True


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### 1-D DYNAMIC PROGRAMMING ####
# --------- 53. Climbing Stairs - Leetcode 70 - Easy ------------
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

    """
    COMPLEXITY:

    The space complexity of the climbStairs function is O(1) because it uses a constant amount of
    extra space to store the variables one, two, and temp.

    The time complexity of the function is O(n) because it iterates n-1 times in the for loop, where
    n is the input parameter. Each iteration takes constant time, so the overall time complexity is
    linear with respect to the input size.
    """

    # one represents the number of ways to get to the nth stair from the (n - 1)st stair via one step
    # remember, we are doing a bottom up approach as opposed to starting from the zeroth stair
    # and working our way up as we already know our 2 bases cases which are:
    # - the number of ways to get to the nth stair from the nth stair is n
    # - the number of ways to get to the nth stair from the (n - 1)st stair is n
    one = 1
    # two represents the number of ways to get to the nth stair from the (n - 1)st stair
    two = 1
    # we start at the 3rd stair since we already know the number of ways to get to the 1st and
    # 2nd stairs
    for _ in range(n - 1):
        # we need to save the number of ways to get to the 3rd stair in a temporary variable
        temp = one
        # the number of ways to get to the 3rd stair is the sum of the number of ways to get to
        one = one + two
        # the number of ways to get to the 2nd stair
        two = temp
    # the number of ways to get to the nth stair
    return one

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#### HEAP OR PRIORITY QUEUE ####
# --------- 55. Kth Largest Element in an Array - Leetcode 215 - Medium ------------
def findKthLargest(nums, k):
    # Initialize a min-heap
    min_heap = []

    # Iterate through the array
    for num in nums:
        # Push the current number onto the heap if the heap size is less than k
        if len(min_heap) < k: heapq.heappush(min_heap, num)
        # If the heap size is k, push the current number and pop the smallest element
        elif num > min_heap[0]: heapq.heapreplace(min_heap, num)

    # The root of the heap is the kth largest element
    return min_heap[0]