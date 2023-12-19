## ARRAYS AND HASHING #####

# --------- 1. Contains Duplicate - Leetcode 217 - Easy ------------
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

    The time complexity of the groupAnagrams function is O(n), where n is the length of the input list strs.
    This is because the function iterates over each element in strs once, performing constant-time operations such as dictionary lookups and updates.
    The worst-case scenario occurs when all the strings in strs are anagrams of each other, resulting in a linear time complexity.

    Therefore, the space complexity and time complexity of the groupAnagrams function are both O(n).
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
