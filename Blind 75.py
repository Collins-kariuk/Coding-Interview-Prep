### LINKED LISTS ###

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# --------- 1. Remove Nth Node from End of List - Leetcode 19 - Medium ------------
def removeNthFromEnd(head, n):
    # Create a dummy node and attach it to the head of the input list.
    dummy = ListNode(val = 0, next = head)
    
    # Initialize 2 pointers, first and second, to point to the dummy node.
    first = dummy
    second = dummy
    
    # Advances first pointer so that the gap between first and second is n nodes apart
    for i in range(n + 1):
        first = first.next
        
    # While the first pointer does not equal null move both first and second to maintain the gap and get nth node from the end
    while (first != None):
        first = first.next
        second = second.next
    
    # Delete the node being pointed to by second.
    second.next = second.next.next
    
    # Return dummy.next
    return dummy.next
    

### TREES ###

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# --------- 2. Invert Binary Tree - Leetcode 19 - Medium ------------
def invertTree(root):
    # check whether the root contains value and if not return none
    if root == None:
        return None

    # swap the left and right children
    # you have to store the left child in a variable first
    # to not lose it
    temp = root.left
    root.left = root.right
    root.right = temp

    # recursively call the method on the right and left children
    invertTree(root.left)
    invertTree(root.right)

    return root
