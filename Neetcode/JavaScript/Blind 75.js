/// ARRAYS & HASHING ///

// --------- 4. Contains Duplicate - Leetcode 217 - Easy ------------

class ContainsDuplicateSolution {
  /**
   * @param {number[]} nums
   * @return {boolean}
   */
  hasDuplicate(nums) {
    let uniqueNums = new Set();

    for (let i = 0; i < nums.length; i++) {
      if (uniqueNums.has(nums[i])) {
        // Use `has` to check for existence in the Set
        return true;
      } else {
        uniqueNums.add(nums[i]); // Add the number to the Set
      }
    }

    return false; // Return false if no duplicates are found
  }
}
