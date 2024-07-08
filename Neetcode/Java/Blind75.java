import java.util.HashSet;
public class Blind75 {
    // ARRAYS & HASHING //
    // --------- 4. Contains Duplicate - Leetcode 217 - Easy ------------
    public boolean containsDuplicate(int[] nums) {
        // Initialize a HashSet that will store the unique elements in nums
        HashSet<Integer> set = new HashSet<>();
        for(int num : nums){
            // If the number already exists in the set, nums contains a duplicate
            if(set.contains(num)) return true;

            // Otherwise, add the number to the set
            set.add(num);
        }
        // We've gone through all numbers in nums and have not found any duplicates
        return false;
    }
}
