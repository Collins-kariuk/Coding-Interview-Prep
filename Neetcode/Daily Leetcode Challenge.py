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
