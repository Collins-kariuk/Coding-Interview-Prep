## ARRAYS AND HASHING #####

# --------- 1. Contains Duplicate - Leetcode 217 - Easy ------------
def containsDuplicate(nums):
    return len(set(nums)) != len(nums)
    # Complexity? O(1) space complexity and O(nlogn) time complexity

    # neetcode's solution - O(n) time and space complexity
    # hashset = set()
    # for n in nums:
        # if n in hashSet:
        #    return True
        # hashset.add(n)
    # return False


# --------- 2. Group Anagrams - Leetcode 49 - Medium ------------
from collections import defaultdict

def groupAnagrams(strs):
    res = defaultdict(list)
    for string in strs:
        count = [0]*26
        for c in string:
            count[ord(c) - ord('a')] += 1
        res[tuple(count)].append(string)
    return res.values()


# --------- 3. Top K Frequent Elements - Leetcode 347 - Medium ------------
def topKFrequent (nums, k):
    count = {}
    # the buckets
    freq = [[] for i in range(len (nums) + 1)]

    for n in nums:
            count[n] = 1 + count.get(n, 0)
    for n, c in count.items():
            freq[c].append(n)

    res = []
    for i in range(len(freq) - 1, 0, -1):
        for n in freq[i]:
            res.append(n)
            if len (res) == k:
                    return res
