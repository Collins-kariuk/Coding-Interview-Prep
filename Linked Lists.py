class Node:
    def __init__(self, dataval = None):
        self.dataval = dataval
        self.nextval = None


class SLinkedList:
    def __init__(self):
        self.headval = None
    
    def listprint(self):
        printval = self.headval
        while printval is not None:
            print(printval.dataval)
            printval = printval.nextval
    
    def atBeginning(self, newdata):
        newNode = Node(newdata)
        # update the new nodes next val to existing node
        newNode.nextval = self.headval
        self.headval = newNode

list1 = SLinkedList()
list1.headval = Node("Mon")
el2 = Node("Tue")
el3 = Node("Wed")
# link first node to second node
list1.headval.nextval = el2
# link second node to third node
el2.nextval = el3

list1.atBeginning("Sun")

list1.listprint()