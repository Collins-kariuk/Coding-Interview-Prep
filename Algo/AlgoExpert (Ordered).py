### HEAPS OR PRIORITY QUEUES ###
# --------- 1. Continuous Median - Hard ------------
# Do not edit the class below except for
# the insert method. Feel free to add new
# properties and methods to the class.
import heapq
class ContinuousMedianHandler:
    def __init__(self):
        self.median = None
        # max heap
        self.small = [] 
        # min heap (default)
        self.large = []

    def insert(self, number):
        # insert into large heap if number is greater than the smallest number in large heap
        if self.large and number > self.large[0]:
            heapq.heappush(self.large, number)
        else:
            heapq.heappush(self.small, -1 * number)

        # balance the heaps if they are not equal in size
        # if small heap is larger than large heap, pop the smallest number from small heap and push it to large heap
        # if large heap is larger than small heap, pop the smallest number from large heap and push it to small heap
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)

    def getMedian(self):
        if len(self.small) > len(self.large):
            self.median = -1 * self.small[0]
        elif len(self.large) > len(self.small):
            self.median = self.large[0]
        else:
            self.median = (-1 * self.small[0] + self.large[0]) / 2
        return self.median

# --------- 2. Continuous Median - Hard ------------
def sortKSortedArray(array, k):
    minHeapWithKElements = heapq.heapify(array[:min(k + 1, len(array))])
    nextIndexToInsertElement = 0

    for idx in range(k + 1, len(array)):
        minElement = heapq.heappop(minHeapWithKElements)
        array[nextIndexToInsertElement] = minElement
        nextIndexToInsertElement += 1
        heapq.heappush(minHeapWithKElements, array[idx])
    
    while len(minHeapWithKElements) != 0:
        minElement = heapq.heappop(minHeapWithKElements)
        array[nextIndexToInsertElement] = minElement
        nextIndexToInsertElement += 1
    
    return array
