# --------- (Dec 31 2023) Largest Substring Between Two Equal Characters - Leetcode 1624 - Easy ------------
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
