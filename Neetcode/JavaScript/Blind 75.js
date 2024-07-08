/// ARRAYS & HASHING ///

// --------- 4. Contains Duplicate - Leetcode 217 - Easy ------------

/**
 * @param {number[]} nums
 * @return {boolean}
 */
function containsDuplicate(nums) {
    // Initialize an object that will store the occurrences of elements in nums
    let dictList = {};
    for (let num of nums) {
        // If the number already exists in the object, nums contains a duplicate
        if (dictList.hasOwnProperty(num)) {
            return true;
        }
        // If the number does not already exist in the object, add it
        else {
            dictList[num] = 1;
        }
    }
    // We've gone through all numbers in nums and have not found any duplicates
    return false;
}