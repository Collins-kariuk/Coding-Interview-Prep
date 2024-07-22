import java.util.PriorityQueue;

public class NeetCode150 {

    // HEAP OR PRIORITY QUEUE //
    // ---------- 1. Kth Largest Element in an Array - Leetcode 215 - Medium -------------
    public int findKthLargest(int[] nums, int k) {
        // Initialize a min-heap (priority queue in Java)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        // Iterate through the array
        for (int num : nums) {
            // Push the current number onto the heap if the heap size is less than k
            if (minHeap.size() < k) minHeap.offer(num);
            // If the heap size is k, push the current number and pop the smallest element
            else if (num > minHeap.peek()) {
                minHeap.poll();
                minHeap.offer(num);
            }
        }

        // The root of the heap is the kth largest element
        return minHeap.peek();
    }
}
