import java.util.HashSet;
import java.util.Stack;
import java.util.HashMap;

public class Blind75 {
    // ARRAYS & HASHING //
    // --------- 1. Contains Duplicate - Leetcode 217 - Easy ------------
    public boolean containsDuplicate(int[] nums) {
        // Initialize a HashSet that will store the unique elements in nums
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            // If the number already exists in the set, nums contains a duplicate
            if (set.contains(num)) return true;
            // Otherwise, add the number to the set
            set.add(num);
        }
        // We've gone through all numbers in nums and have not found any duplicates
        return false;
    }

    // --------- 2. Valid Anagram - Leetcode 242 - Easy ------------
    public boolean isAnagram(String s, String t) {
        // If the lengths of the strings are not equal, they can't be anagrams
        if (s.length() != t.length()) return false;

        // Initialize 2 HashMaps to store the number of occurrences of each character
        HashMap<Character, Integer> countS = new HashMap<>();
        HashMap<Character, Integer> countT = new HashMap<>();

        // Loop through the strings and count the number of occurrences of each character
        for (int i = 0; i < s.length(); i++) {
            char charS = s.charAt(i);
            char charT = t.charAt(i);

            countS.put(charS, countS.getOrDefault(charS, 0) + 1);
            countT.put(charT, countT.getOrDefault(charT, 0) + 1);
        }

        // Compare the two HashMaps
        return countS.equals(countT);
    }

    public boolean isAnagramB(String s, String t) {
        // Create a function that converts a string to a HashMap with character counts
        HashMap<Character, Integer> sDict = dictify(s);
        HashMap<Character, Integer> tDict = dictify(t);

        // Compare the two HashMaps
        return sDict.equals(tDict);
    }

    private HashMap<Character, Integer> dictify(String s) {
        // Create a HashMap to store the character counts
        HashMap<Character, Integer> charCounter = new HashMap<>();
        // Convert the string to an array of characters and count the number of occurrences of each
        // character
        for (char c : s.toCharArray()) {
            // If the character already exists in the HashMap, increment the count
            if (charCounter.containsKey(c)) {
                charCounter.put(c, charCounter.get(c) + 1);
            } else {
                // Otherwise, add the character to the HashMap with a count of 1
                charCounter.put(c, 1);
            }
        }
        return charCounter;
    }

    // --------- 3. Two Sum - Leetcode 1 - Easy ------------
    public int[] twoSum(int[] nums, int target) {
        // HashMap to store already visited numbers in nums alongside their indices as key-value
        // pairs
        HashMap<Integer, Integer> twoSumMap = new HashMap<>();

        // Looping through nums, where each number in each iteration will act as num1
        for (int i = 0; i < nums.length; i++) {
            // The second number, num2, that when added to num1 will produce target
            int num2 = target - nums[i];
            // When num2 is already in the HashMap, it means we've already found our 2 numbers and
            // we can return their indices as an array
            if (twoSumMap.containsKey(num2)) {
                return new int[] { i, twoSumMap.get(num2) };
            }
            // Otherwise we add the number at the current iteration into the HashMap which will
            // later serve as num2
            twoSumMap.put(nums[i], i);
        }
        // In case there is no solution, though the problem statement guarantees one
        throw new IllegalArgumentException("No two sum solution");
    }

    // TWO POINTERS //
    // --------- 4. Valid Palindrome - Leetcode 125 - Easy ------------
    // Method to check if a character is alphanumeric
    public boolean alphanum(char c) {
        return (('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ('0' <= c && c <= '9'));
    }

    // Method to check if a string is a palindrome
    public boolean isPalindrome(String s) {
        // The left and right pointers
        int l = 0;
        int r = s.length() - 1;

        while (l < r) {
            // Skip non-alphanumeric characters
            while (l < r && !alphanum(s.charAt(l))) {
                l++;
            }
            while (l < r && !alphanum(s.charAt(r))) {
                r--;
            }

            // Check whether the characters at the left and right pointers are equal
            // Return false immediately if they are not
            if (Character.toLowerCase(s.charAt(l)) != Character.toLowerCase(s.charAt(r))) {
                return false;
            }

            l++;
            r--;
        }

        return true;
    }

    // --------- 5. Container With Most Water - Leetcode 11 - Medium ------------
    public int maxArea(int[] heights) {
        // Initialize pointers
        int l = 0;
        int r = heights.length - 1;
        // Initialize variable that will store the maximum volume
        int res = 0;

        // Continue with the loop as long as the pointers do not cross each other
        while (l < r) {
            // Calculate the current area at the specific point in the iteration using the basic
            // equation of base * height. Here, the base is the difference between the pointers,
            // and the height is the smaller of the two values at the left and right pointers
            int currArea = (r - l) * Math.min(heights[l], heights[r]);
            // The current maximum volume is the larger of the previous volume and the current
            // volume
            res = Math.max(res, currArea);
            // When the height at the left pointer is smaller than the height at the right pointer
            // we increment the left pointer by one so as to still preserve the bigger height at
            // the right pointer since that height may be the smaller of 2 heights later in the
            // iteration
            if (heights[l] < heights[r]) {
                l++;
            } else {
                r--;
            }
        }
        return res;
    }

    // STACK //
    // --------- 6. Valid Parentheses - Leetcode 20 - Easy ------------
    public static boolean isValid(String s) {
        // Stack to store (potentially matching) open parentheses
        Stack<Character> stack = new Stack<>();
        // Map with closing to open parentheses as key-value pairs
        HashMap<Character, Character> closeToOpen = new HashMap<>();
        closeToOpen.put(')', '(');
        closeToOpen.put(']', '[');
        closeToOpen.put('}', '{');

        for (char c : s.toCharArray()) {
            // When char is a closing parenthesis
            if (closeToOpen.containsKey(c)) {
                // If the stack is not empty and the open parenthesis at the top of the stack
                // matches the corresponding open parenthesis for the current closing parenthesis,
                // it indicates that we have a matching pair of parentheses. In this case, we can
                // remove the open parenthesis from the top of the stack.
                if (!stack.isEmpty() && stack.peek() == closeToOpen.get(c)) {
                    stack.pop();
                } 
                // If the stack is empty or if the open parenthesis at the top of the stack does
                // not match the corresponding open parenthesis for the current closing
                // parenthesis, the input string is considered invalid.
                //
                // In the case where the stack is empty, a sample input string might be
                // '(()))[]{}'. By the time we reach the third closing parenthesis ')', the stack
                // will be empty because two '(' characters will have been popped in previous
                // iterations.
                //
                // In the case where the top of the stack does not match the expected open
                // parenthesis, a sample string could be '[{]}'. Here, the stack will not be empty,
                // but when we reach the closing bracket ']', the top of the stack will be '{',
                // which does not match the expected counterpart for ']'.
                else {
                    return false;
                }
            } 
            // When char is an open parenthesis, we add it to the stack and it will be compared in
            // a later iteration
            else {
                stack.push(c);
            }
        }
        // The input string will be valid only if the stack is empty after iterating through the
        // entire string. This means that all matching parentheses have been successfully paired
        // and removed, ensuring they appear in the correct order.
        return stack.isEmpty();
    }

    // BINARY SEARCH //
    // --------- 7. Find Minimum in Rotated Sorted Array - Leetcode 153 - Medium ------------
    public int findMin(int[] nums) {
        // Variable that'll store the current minimum
        int res = nums[0];
        // Initialize the pointers
        int l = 0;
        int r = nums.length - 1;

        while (l <= r) {
            // If the number at the left pointer is less than the number at the right pointer, it
            // indicates that the array is already sorted. We can then safely return the number at
            // the left pointer or the current minimum, whichever is smaller.
            if (nums[l] < nums[r]) {
                res = Math.min(res, nums[l]);
                break;
            }

            // Calculate the middle pointer
            int mid = (l + r) / 2;
            // Before further comparison, the number at the middle pointer will serve as the
            // minimum
            res = Math.min(res, nums[mid]);
            // If the middle element is greater than or equal to the element at the left pointer,
            // it indicates that the left segment of the sublist is already sorted. Due to the
            // array's rotation, searching in the left segment is illogical, as it will always
            // contain larger values compared to the right segment. Therefore, our search should
            // concentrate on the right segment of the array.
            if (nums[mid] >= nums[l]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return res;
    }

    // LINKED LIST //
    // Definition for singly-linked list.
    class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
        }
    }

    // --------- 8. Reverse Linked List - Leetcode 209 - Easy ------------
    public static ListNode reverseList(ListNode head) {
        // Initialize the 2 needed pointers required to traverse through the linked list
        ListNode prev = null;
        ListNode curr = head;

        // Continue with the loop as long as the current pointer does not point at a null node
        while (curr != null) {
            // Save a reference to the node following the current one, as it will become the
            // current node in the next iteration.
            ListNode placeholder = curr.next;
            // Changing the direction of the pointer for the current node.
            curr.next = prev;
            // Advancing the previous pointer to the location of the current pointer
            prev = curr;
            // Advancing the current pointer to the location of the placeholder we conveniently
            // saved
            curr = placeholder;
        }
        // The new head of the reversed linked list will be the node that 'prev' is pointing to
        return prev;
    }

    // --------- 9. Remove Nth Node from End of List - Leetcode 19 - Medium ------------
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // Create a dummy node that points to the head of the list
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode left = dummy;
        ListNode right = head;

        // Move the right pointer n steps ahead
        while (n > 0) {
            right = right.next;
            n--;
        }

        // Move both pointers until the right pointer reaches the end
        // The left pointer will then be at the node before the one to be removed
        while (right != null) {
            left = left.next;
            right = right.next;
        }

        // Remove the nth node from the end
        left.next = left.next.next;

        // Return the head of the modified list
        return dummy.next;
    }
}
