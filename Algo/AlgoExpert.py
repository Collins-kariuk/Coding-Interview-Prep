#------7. Branch Sum---------
# This is the class of the input root. Do not edit it.
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
def branchSums(root):
    sum = 0
    res = []
    while (root is not None):
        sum += root.val
        if root.left is None and root.right is not None:
            branchSums(root.right)
        if root.left is not None and root.right is None:
            branchSums(root.left)
        else:
            branchSums(root.left)
            branchSums(root.right)
        res.append(sum)
    return res

#------ 3. Sorted Square Array ---------
def sortedSquaredArray(array):
    sortedSquares = [0 for _ in array]
    smallerValueIdx = 0
    largerValueIdx = len (array) - 1
    for idx in (range(len(array))):
        smallerValue = array[smallerValueIdx]
        largerValue = array[largerValueIdx]
        if abs(smallerValue) > abs(largerValue):
            sortedSquares[idx] = smallerValue * smallerValue
            smallerValueIdx += 1 
        else:
            sortedSquares[idx] = largerValue * largerValue
            largerValueIdx -= 1
    return sortedSquares

#------4. Tournament Winner---------
def tournamentWinner(competitions, results):
    # records the team and their scores in key:value format
    scoreRecord = {}
    # loops through each competition sublist pair and make sure that every competitor
    # is in the score records
    for competition in competitions:
        # checks if the first competitor at each competition is in the score records
        # if not, then we add them to the score records with an initial score of 0 
        if competition[0] not in scoreRecord:
            scoreRecord[competition[0]] = 0
        # checks if the second competitor at each competition is in the score records
        # if not, then we add them to the score records with an initial score of 0
        if competition[1] not in scoreRecord:
            scoreRecord[competition[1]] = 0

    # loop through each competition results and update the score of each competitor
    for i in range(len(results)):
        # if the result is 0 (indicating the away team won), then we add 3 points
        # to the corresponding away team
        if results[i] == 0:
           scoreRecord[competitions[i][1]] += 3
        # if the result is 1 (indicating the home team won), then we add 3 points
        # to the corresponding home team
        if results[i] == 1:
            scoreRecord[competitions[i][0]] += 3

    # list of all scores of the teams involved
    teamScores=list(scoreRecord.values())
    # getting the highest score
    highestScore=max(teamScores)
    # looping through the keys (teams) in the score record
    for team in scoreRecord:
        # if the value (score) of the team is the highest score, we simply return
        # that team.
        if scoreRecord[team]==highestScore:
            return team

#------12. Class Photos---------
def classPhotos(redShirtHeights, blueShirtHeights):
    checker = 0
    for i in range(len(redShirtHeights)):
        if redShirtHeights[i] == blueShirtHeights[i]:
            return False
        if redShirtHeights[i] > blueShirtHeights[i]:
            checker += 1
    print(checker)
    return (checker == 0 or checker == len(redShirtHeights))

#------13. Tandem Bicycle---------
def tandemBicycle(redShirtSpeeds, blueShirtSpeeds, fastest):
    # dealing with the minimum
    redShirtSpeeds.sort()
    blueShirtSpeeds.sort()
    minimum = 0
    minimumPairs = []
    for i in range(len(redShirtSpeeds)):
        minimum+=max([redShirtSpeeds[i], blueShirtSpeeds[i]])
        # minimumPairs.append([redShirtSpeeds[i], blueShirtSpeeds[i]])
    
    for pair in minimumPairs:
        minimum += max(pair)

    # dealing with the maximum
    reverseArrayInPlace(blueShirtSpeeds)
    maximumPairs = []
    maximum = 0
    for i in range(len(redShirtSpeeds)):
        maximum+=max([redShirtSpeeds[i], blueShirtSpeeds[i]])
        # maximumPairs.append([redShirtSpeeds[i], blueShirtSpeeds[i]])
    
    for pair in maximumPairs:
        maximum += max(pair)

    if fastest:
        return maximum
    return minimum
def reverseArrayInPlace(array):
    start, end = 0, len(array) - 1
    while start < end:
        array[start], array[end] = array[end], array[start]
        start += 1
        end -= 1

#------16. Nth Fibonacci---------
def getNthFib(n):
    if n==1 or n==0:
        return 0
    res = [0, 1]
    for i in range(n-2):
        a = res[i]+res[i+1]
        res.append(a)
    return res

#------------------- 17. Product Sum ------------------
def productSum(array, multiplier = 1):
    summation = 0
    for el in array:
        if type(el) is list:
            summation += productSum(el, multiplier + 1)
        else: 
            summation += el
    return summation * multiplier

#------------------- 18. Find Three Largest Numbers ------------------
import heapq
def findThreeLargestNumbers(array):
    negArray = [-num for num in array]
    res = []
    heapq.heapify(negArray)
    for i in range(3):
        res.append(-heapq.heappop(negArray))
    return res[::-1]


# -------------- Common Characters --------------
def commonCharacters(strings):
    for i in range(len(strings)):
        strings[i] = list(set(strings[i]))
    res = []
    counter = {}
    for i in range(len(strings)):
        for j in range(len(strings[i])):
            if strings[i][j] in counter:
                counter[strings[i][j]] += 1
            else:
                counter[strings[i][j]] = 1
    for key in counter:
        if counter[key] == len(strings):
            res.append(key)
    return res

# -------------- Generate Document --------------
def generateDocument(characters, document):
    docList = list(document)
    for c in characters:
        if c in docList:
            docList.remove(c)
    return len(docList) == 0


# -------------- First Non-Repeating Character --------------
def firstNonRepeatingCharacter(string):
    counter = {}
    for c in string:
        if c in counter:
            counter[c] += 1
        else:
            counter[c] = 1
    print(counter)
    countOnes = []
    for key in counter:
        if counter[key] == 1:
            countOnes.append(key)
    indices = []

    for key in countOnes:
        indices.append(string.index(key))
    if len(indices) == 0:
        return -1
    return min(indices)


# -------------- Run-Length Encoding --------------
def runLengthEncoding(string):
    res = ""
    l = 0
    r = 1
    while r < len(string):
        if string[r] == string[l]:
            r += 1
        else:
            difference = r - l
            if difference > 9:
                trunc = difference // 9
                mod = difference % 9
                combo = ('9' + string[l]) * trunc + str(mod) + string[l]
                res += combo
            else:
                res += str(difference) + string[l]
            l = r
            r += 1
            
    if r - l >= 9:
        # print(r - l)
        trunc = (r - l) // 9
        # print(trunc)
        mod = (r - l) % 9
        # print(mod)
        combo = ('9' + string[l]) * trunc + str(mod) + string[l]
        res += combo
    else:
        res += str(r - l) + string[l]
    return res


# -------------- Semordnilap --------------
def reversedWord(word):
    wordList = list(word)
    reversedWordList = wordList[::-1]
    joinedReversedWord = "".join(reversedWordList)
    return joinedReversedWord

def semordnilap(words):
    res = []
    for word in words:
        wordRev = reversedWord(word)
        if wordRev in words and wordRev != word:
            res.append([word, wordRev])
            words.remove(wordRev)
    return res


# -------------- Transpose Matrix (WRONG) --------------
def transposeMatrix(matrix):
    res = [[0] * len(matrix)] * len(matrix)
    print(res)
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            res[j][i] = matrix[i][j]
    return res


# -------------- Caesar Cipher Encryptor --------------
def caesarCipherEncryptor(string, key):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    res = ""
    for char in string:
        charIndexInAlphabet = alphabet.index(char)
        finalPos = (charIndexInAlphabet + key) % 26
        res += alphabet[finalPos]
    return res

# -------------- Caesar Cipher Encryptor --------------
def merge(intervals):
    res = []
    for i in range(len(intervals)-1):
        if intervals[i + 1][0] in range(intervals[i][0], intervals[i][1] + 1):
            res.append([intervals[i][0], intervals[i + 1][1]])
    return res


print(merge([[1, 4], [4, 5]]))
print(merge([[1, 3], [2, 6], [8, 10], [15, 18]] ))
