# --------- 1. Median of Two Sorted Arrays - Leetcode 4 - Hard ------------
# There are two sorted arrays nums1 and nums2 of size m and n respectively.
# Find the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).
def findMedianSortedArrays(nums1, nums2):
    A, B = nums1, nums2
    # A is the shorter array
    # B is the longer array
    total = len(nums1) + len(nums2)
    # total is the total length of the two arrays
    half = total // 2

    # if A is longer than B, swap them
    if len(B) < len(A):
        A, B = B, A
    
    # perform binary search on A
    l, r = 0, len(A) - 1
    while True:
        # i is the index of the middle element of A
        i = (l + r) // 2
        # j is the index of the middle element of B
        # partition A and B such that the number of elements in the left partition
        # is equal to the number of elements in the right partition
        j = half - i - 2

        # Aleft is the left element of A
        # Aright is the right element of A
        Aleft = A[i] if i >= 0 else float("-infinity")
        Aright = A[i + 1] if (i + 1) < len(A) else float("infinity")
        # Bleft is the left element of B
        # Bright is the right element of B
        Bleft = B[j] if j >= 0 else float("-infinity")
        Bright = B[j + 1] if (j + 1) < len(B) else float("infinity")

        # if the partition is correct, return the median
        if Aleft <= Bright and Bleft <= Aright:
            # if the total length is odd, return the middle element
            if total % 2:
                return min(Aright, Bright)
            # if the total length is even, return the average of the middle two elements
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        # if the partition is not correct, move the partition
        # in this case, the left partition of A is too big
        # decrease A's left partition
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1
