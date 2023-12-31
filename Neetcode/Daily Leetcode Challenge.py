# --------- (Dec 31 2023) Largest Substring Between Two Equal Characters - Leetcode 1624 - Easy ------------
import math


def maxLengthBetweenEqualCharacters(s):
    """
    COMPLEXITY: O(n) time | O(n) space

    DESCRIPTION: 
    The space complexity of this function is O(n), where n is the length of the input string.
    This is because the function uses a dictionary to store the index of each character,
    and the size of the dictionary will grow linearly with the size of the input string.
    The time complexity of this function is O(n), where n is the length of the input string.
    This is because the function iterates through the string once, performing constant-time
    operations for each character. The maximum number of iterations is equal to the length
    of the string, resulting in a linear time complexity.

    NOTES: Even if a certain character has more than 2 occurrences, the function will only
    consider the first and last occurrences of that character. 
    """

    # Create a dictionary to store the index of each character (key: character, value: index)
    d = {}
    # Create a variable to store the max length
    max_length = -1
    # Iterate through the string
    for i, c in enumerate(s):
        # If the character is not in the dictionary, it means it's the first occurrence of
        # that character. In this case, the function adds the character to the dictionary
        # with its index as the value.
        if c not in d:
            d[c] = i
        # If the character is already in the dictionary, it means it's a repeated occurrence
        # of that character. In this case, the function calculates the length between the
        # current index i and the index stored in the dictionary for that character. It then
        # updates the max_length variable if the calculated length is greater than the current max_length.
        else:
            max_length = max(max_length, i - d[c] - 1)
    # Return the max length
    return max_length


# --------- (Jan 1 2024) Assign Cookies - Leetcode 455 - Easy ------------
def findContentChildren(g, s):
    """
    COMPLEXITY: O(nlogn) time | O(1) space

    DESCRIPTION: 
    The space complexity of this function is O(1), because the function does not use any
    additional data structures to store information.

    The time complexity of this function is O(nlogn), where n is the length of the input
    array g. This is because the function sorts the input array g, which takes O(nlogn) time.
    The function then iterates through the input array s, performing constant-time operations
    for each element. The maximum number of iterations is equal to the length of the input
    array s, resulting in a linear time complexity.

    NOTES: The function finds the maximum number of content children by greedily assigning
    cookies to children with the smallest greed factors. It does not assume any specific
    order of the input arrays g and s.
    """

    # Sort the input arrays g and s
    g.sort()
    s.sort()

    # Create a variable to store the number of content children
    count = 0
    i = 0  # Index for array g
    j = 0  # Index for array s

    while i < len(g) and j < len(s):
        # If the current cookie j is greater than or equal to the greed factor of the
        # current child i, the function gives the cookie to the child and moves to the
        # next child
        if s[j] >= g[i]:
            count += 1
            i += 1
        # we move to the next cookie regardless of whether the current child gets a cookie because we want to
        # find the smallest cookie that can satisfy the current child and we can only do that by checking all cookies
        # in the sorted array s
        j += 1

    # Return the number of content children
    return count


# --------- (Jan 2 2024) Convert an Array into a 2D Array with Conditions - Leetcode 2610 - Medium ------------
def dictify(nums):
    countNums = {}
    for num in nums:
        if num in countNums:
            countNums[num] += 1
        else:
            countNums[num] = 1
    return countNums


def findMatrix(nums):
    """
    COMPLEXITY: O(n) time | O(n) space

    DESCRIPTION:
    The space complexity of the findMatrix function is O(n), where n is the length of the input array nums.
    This is because the function creates a result list res with a number of sublists equal to the maximum number of
    occurrences of a number in the input array. The size of each sublist will depend on the number of unique elements
    in the input array.

    The time complexity of the findMatrix function is O(n^2), where n is the length of the input array nums.
    This is because the function iterates through each element in the input array and checks if it is already present
    in any of the sublists in the result. The worst-case scenario occurs when all elements in the input array are unique,
    resulting in nested loops that iterate through all sublists for each element.

    NOTES: The function creates a result list with a number of sublists equal to the maximum number of occurrences of a
    number in the input array. It then iterates through the input array and adds each element to the first sublist that
    does not contain it. This ensures that the number of unique elements in each sublist is equal to the maximum number
    of occurrences of a number in the input array.
    """

    # the number of sublists in the result is equal to the maximum number of occurrences of a number in the input array
    countNumsVals = list(dictify(nums).values())
    numResSublists = max(countNumsVals)
    res = [[] for _ in range(numResSublists)]

    for num in nums:
        # if the number is not in any of the sublists in the result, add it to the first sublist
        # if the number is already in a sublist in the result, add it to the next sublist, and so on
        for sublist in res:
            if num not in sublist:
                sublist.append(num)
                # we have to break out of the loop after adding the number to the first sublist that does not contain it
                # otherwise, the number will be added to all sublists that do not contain it
                break
    return res


# --------- (Jan 3 2024) Number of Laser Beams in a Bank - Leetcode 2125 - Medium ------------
def numberOfBeams(bank):
    """
    COMPLEXITY:
    The space complexity of the numberOfBeams function is O(n), where n is the number of rows in the input array bank.
    This is because the function creates a list to store the number of devices in each row, a list to store the number
    of laser beams in between each row pair, and a list to store the number of laser beams in each row. The size of each
    list will depend on the number of rows in the input array.

    The time complexity of the numberOfBeams function is O(n^2), where n is the number of rows in the input array bank.
    This is because the function iterates through each row in the input array and then iterates through each space in
    the row. The worst-case scenario occurs when all spaces in the input array contain devices, resulting in nested loops
    that iterate through all spaces in each row.

    NOTES: The function first iterates through each row in the input array and counts the number of devices in each row.
    It then iterates through the list of device counts and calculates the number of laser beams in between each row pair.
    The function returns the sum of the number of laser beams in the bank.
    """

    # initialize a list to store the number of devices in each row
    deviceCounts = []
    numDevices = 0
    # iterate through each row in the bank
    for row in bank:
        for space in row:
            # if the space contains a device, increment the number of devices in the row
            if space == '1':
                numDevices += 1
        # after iterating through all spaces in the row, append the number of devices to the list
        deviceCounts.append(numDevices)
        # reset the number of devices to 0 for the next row
        numDevices = 0

    # initialize a list to store the number of laser beams in each row
    rowsWithDevices = []
    for deviceCount in deviceCounts:
        # if the row contains at least one device, add it to the list
        # we are not interested in rows that do not contain any devices
        if deviceCount != 0:
            rowsWithDevices.append(deviceCount)

    # initialize a list to store the number of laser beams in between each row pair
    laserBeamsPerRow = []
    for i in range(len(rowsWithDevices) - 1):
        # the number of laser beams between two rows is equal to the product of the number of devices in any two adjacent rows
        laserBeamsPerRow.append(rowsWithDevices[i] * rowsWithDevices[i + 1])
    # return the sum of the number of laser beams in the bank
    return sum(laserBeamsPerRow)

# --------- (Jan 4 2024) Minimum Number of Operations to Make Array Empty - Leetcode 2870 - Medium ------------


def dictify(nums):
    counter = {}
    for num in nums:
        if num in counter:
            counter[num] += 1
        else:
            counter[num] = 1
    return counter


def minOperations(nums):
    """
    COMPLEXITY:
    The space complexity of the minOperations function is O(n), where n is the length of the input array nums.
    This is because the function creates a dictionary to store the number of occurrences of each element in the input array.
    The size of the dictionary will depend on the number of unique elements in the input array.

    The time complexity of the minOperations function is O(n), where n is the length of the input array nums.
    This is because the function iterates through each element in the input array and performs constant-time operations
    for each element. The maximum number of iterations is equal to the length of the input array, resulting in a linear
    time complexity.
    """

    # create a dictionary to store the number of occurrences of each element in the input array
    counter = dictify(nums)
    # create a list to store the number of operations required to empty the input array
    frequencies = list(counter.values())
    res = 0
    for num in frequencies:
        # when the number is equal to 1 we cannot reduce it to groups of 2 or 3, so we return -1
        if num == 1:
            return -1
        # when the number is greater than 1, we can reduce it to groups of 2 or 3
        # we do this by dividing the number by 3 and rounding up to the nearest integer regardless of whether the number
        # is divisible by 2 or 3
        # this is because any number greater than 1 can be reduced to groups of 2 or 3
        res += math.ceil(num / 3)
    return res


# --------- (Jan 5 2024) Longest Increasing Subsequence - Leetcode 300 - Medium ------------
def lengthOfLIS(nums):
    """
    COMPLEXITY:
    The space complexity of the lengthOfLIS function is O(n), where n is the length of the input array nums.
    This is because the function creates a list to store the length of the longest increasing subsequence ending at each index.
    The size of the list will depend on the length of the input array.

    The time complexity of the lengthOfLIS function is O(n^2), where n is the length of the input array nums.
    This is because the function iterates through the input array twice, performing constant-time operations for each element.
    The maximum number of iterations is equal to the length of the input array, resulting in a quadratic time complexity.

    NOTES: There exists a more efficient solution to this problem that uses binary search to reduce the time complexity to O(nlogn).
    However, the solution below is easier to understand and implement. There also exists a brute force solution that uses recursion
    to generate all possible subsequences and then checks if each subsequence is increasing (via DFS). This solution has a time
    complexity of O(2^n) and is not included below.
    """

    # create a list to store the length of the longest increasing subsequence ending at each index
    LIS = [1] * len(nums)
    # iterate in reverse order
    for i in range(len(nums) - 1, -1, -1):
        # iterate through the input array starting from the index after the current index
        # essentially looking at the numbers to the right of the current number
        for j in range(i + 1, len(nums)):
            # there can only be a valid increasing subsequence from the current index to the next index if the current
            # number is less than the next number
            if nums[i] < nums[j]:
                # the length of the longest increasing subsequence ending at the current index is equal to the maximum
                # of the current length and the length of the longest increasing subsequence ending at the next index
                # in the example, [1, 2, 4, 3], if we want to find the length of the longest increasing subsequence at
                # index 2 (4), we have the choice of only including the 4 or including the 4 and the longest increasing
                # subseqence at index 3 (3)
                # however, since 4 > 3, it means that no valid increasing subsequence can be formed by including the 4
                # and the 3, so we only include the 4 which is why we need the if check
                LIS[i] = max(LIS[i], 1 + LIS[j])
    # return the maximum length of the longest increasing subsequence
    return max(LIS)


# --------- (Jan 6 2024) Maximum Profit in Job Scheduling - Leetcode 1235 - Hard ------------
import bisect
def jobScheduling(startTime, endTime, profit):
    """
    
    COMPLEXITY:
    The space complexity of the jobScheduling function is O(n), where n is the length of the input arrays startTime, endTime, and profit.
    This is because the function creates a dictionary to store the maximum profit for each index in the input arrays.
    The size of the dictionary will depend on the length of the input arrays.

    The time complexity of the jobScheduling function is O(nlogn), where n is the length of the input arrays startTime, endTime, and profit.
    This is because the function iterates through the input arrays and performs constant-time operations for each element.
    The maximum number of iterations is equal to the length of the input arrays, resulting in a linear time complexity.
    The function also uses the bisect module to perform binary search, which has a time complexity of O(logn).
    
    NOTES: The function first sorts the input arrays by the end time of each job. It then iterates through the input arrays
    and uses binary search to find the index of the next job that starts after the current job ends. The function then
    calculates the maximum profit for the current job by adding the profit of the current job to the maximum profit of the
    next job. The function returns the maximum profit of the last job.
    """

    # sort the input arrays by the start time of each job
    intervals = sorted(zip(startTime, endTime, profit))
    # create a dictionary to store the maximum profit for each index in the input arrays
    cache = {}

    def dfs(i):
        # when i is equal to the length of the input arrays, it means we have reached the end of the intervals
        if i == len(intervals):
            return 0
        # if the index is already in the dictionary, it means we have already calculated the maximum profit for that index
        # so we return the maximum profit for that index
        if i in cache:
            return cache[i]
        
        # don't include the element at index i
        res = dfs(i + 1)

        # include the element at index i
        # we use binary search to find the index of the next job that starts after the current job ends
        # we use the bisect module to perform binary search
        j = bisect.bisect(intervals, (intervals[i][1], -1, -1))

        # the maximum profit for the current job is equal to the profit of the current job plus the maximum profit of the next job
        # we use max to ensure that we are always returning the maximum profit
        cache[i] = res = max(res, intervals[i][2] + dfs(j))
        return res
    
    return dfs(0)
