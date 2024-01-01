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
