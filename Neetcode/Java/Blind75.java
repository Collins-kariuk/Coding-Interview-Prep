import java.util.HashSet;
import java.util.HashMap;
public class Blind75 {
    // ARRAYS & HASHING //
    // --------- 1. Contains Duplicate - Leetcode 217 - Easy ------------
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
}
