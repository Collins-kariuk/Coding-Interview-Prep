# --------- Merge Strings Alternately - Leetcode 1768 - Easy ------------
def mergeAlternately(word1, word2):
    l = 0
    r = 0
    res = []
    while l < len(word1) or r < len(word2):
        if l < len(word1):
            res.append(word1[l])
            l += 1
        if r < len(word2):
            res.append(word2[r])
            r += 1
    return ''.join(res)


# --------- Greatest Common Divisor of Strings - Leetcode 1071 - Easy ------------
def gcdOfStrings(str1, str2):
    len1 = len(str1)
    len2 = len(str2)

    def isDivisor(l):
        if len1 % l != 0 or len2 % l != 0:
            return False
        factor1 = len1 // l
        factor2 = len2 // l
        return str1[:l] * factor1 == str1 and str1[:l] * factor2 == str2
    
    for l in range(min(len1, len2), 0, -1):
        if isDivisor(l):
            return str1[:l]
    return ""


# --------- Delete the Middle Node of a Linked List - Leetcode 2095 - Medium ------------
def deleteMiddle(head):
    pass

# print(1 // 2)

for i in range(-1):
    print(i)

