"""
evaluation_dataset.py
Modular test suite containing comprehensive ground-truth Q&A pairs 
for evaluating the CTCI RAG pipeline.
"""

def get_test_dataset():
    return [
        {
            "question": "Write code to reverse a C-Style String",
            "ground_truth": (
                "This is a classic interview question. The key is to reverse the string in place "
                "while correctly handling the null character (\\0). Use two pointers: one pointing to the "
                "start of the string and another advancing to the end of the string (just before \\0). "
                "Then swap characters while str < end, incrementing str and decrementing end.\n\n"
                "C Code Implementation:\n"
                "void reverse(char *str) {\n"
                "    char *end = str;\n"
                "    char tmp;\n"
                "    if (str) {\n"
                "        while (*end) {\n"
                "            ++end;\n"
                "        }\n"
                "        --end;\n"
                "        while (str < end) {\n"
                "            tmp = *str;\n"
                "            *str++ = *end;\n"
                "            *end-- = tmp;\n"
                "        }\n"
                "    }\n"
                "}"
            )
        },
        {
            "question": "How do you check if a string has all unique characters without additional data structures?",
            "ground_truth": (
                "If no additional data structures can be used, you can compare every character in the string "
                "to every other character using a nested loop. This takes O(N^2) time complexity and O(1) space complexity. "
                "Alternatively, if you are allowed to modify the string, you can sort it first in O(N log N) time and check adjacent characters.\n\n"
                "Code Implementation (Nested Loop approach):\n"
                "bool isUniqueChars(string str) {\n"
                "    for (int i = 0; i < str.length(); i++) {\n"
                "        for (int j = i + 1; j < str.length(); j++) {\n"
                "            if (str[i] == str[j]) {\n"
                "                return false;\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    return true;\n"
                "}"
            )
        },
        {
            "question": "Describe an algorithm to set row and column to 0 if an element in an MxN matrix is 0",
            "ground_truth": (
                "To achieve O(1) auxiliary space instead of using separate boolean arrays for rows and columns, "
                "you can use the first row and first column of the matrix itself as marker storage.\n\n"
                "Algorithm steps:\n"
                "1. Check if the first row and first column contain any zeros and store flags.\n"
                "2. Iterate through the rest of the matrix; if matrix[i][j] == 0, set matrix[i][0] = 0 and matrix[0][j] = 0.\n"
                "3. Iterate through the matrix again, using the first row/col markers to set internal cells to 0.\n"
                "4. Finally, zero out the first row and column based on initial flags."
            )
        },
        {
            "question": "How do you remove duplicates from an unsorted linked list without a temporary buffer?",
            "ground_truth": (
                "If a temporary buffer (like a HashSet) is not allowed, we cannot keep track of previously seen elements in O(1) lookup time. "
                "Instead, we can use two pointers: 'current' to iterate through each node, and 'runner' to iterate ahead and check for duplicates, "
                "deleting any node whose value matches current. This operates in O(N^2) time and O(1) space.\n\n"
                "Code Implementation:\n"
                "void deleteDups(LinkedListNode head) {\n"
                "    LinkedListNode current = head;\n"
                "    while (current != null) {\n"
                "        LinkedListNode runner = current;\n"
                "        while (runner.next != null) {\n"
                "            if (runner.next.data == current.data) {\n"
                "                runner.next = runner.next.next;\n"
                "            } else {\n"
                "                runner = runner.next;\n"
                "            }\n"
                "        }\n"
                "        current = current.next;\n"
                "    }\n"
                "}"
            )
        },
        {
            "question": "Design a stack with min function returning the minimum element in O(1) time.",
            "ground_truth": (
                "To support a min() operation in O(1) time without scanning the stack, maintain an auxiliary stack "
                "(minStack) that tracks the minimum values. When pushing an element, push it onto the main stack, and push it onto minStack "
                "only if it is less than or equal to the current minimum. When popping, if the popped value equals the top of minStack, pop minStack as well.\n\n"
                "Code Logic:\n"
                "void push(int value) {\n"
                "    if (value <= min()) {\n"
                "        minStack.push(value);\n"
                "    }\n"
                "    stack.push(value);\n"
                "}\n"
                "int pop() {\n"
                "    int value = stack.pop();\n"
                "    if (value == min()) {\n"
                "        minStack.pop();\n"
                "    }\n"
                "    return value;\n"
                "}"
            )
        },
        {
            "question": "Check if s2 is a rotation of s1 using only one call to isSubstring.",
            "ground_truth": (
                "The brilliant insight is that if s2 is a rotation of s1, then s2 is always a substring of (s1 + s1).\n\n"
                "Code Implementation:\n"
                "bool isRotation(string s1, string s2) {\n"
                "    if (s1.length() == s2.length() && s1.length() > 0) {\n"
                "        string s1s1 = s1 + s1;\n"
                "        return isSubstring(s1s1, s2);\n"
                "    }\n"
                "    return false;\n"
                "}"
            )
        },
        {
            "question": "Find the intersecting node of two singly linked lists by reference.",
            "ground_truth": (
                "To find the intersection node by reference without extra data structures:\n"
                "1. Run through both linked lists to get their lengths and tails.\n"
                "2. If the tails do not match by reference, the lists do not intersect (return null).\n"
                "3. Set two pointers at the start of each list. Advance the pointer for the longer list by the difference in lengths.\n"
                "4. Traverse both lists simultaneously until the pointer references match."
            )
        },
        {
            "question": "Check if a binary tree is balanced where heights differ by at most 1.",
            "ground_truth": (
                "To prevent an inefficient O(N log N) or O(N^2) top-down height check, implement a bottom-up approach "
                "where the recursive height function checks balance and returns an error code (-1) if any subtree is unbalanced, "
                "allowing immediate short-circuiting.\n\n"
                "Code Logic:\n"
                "int checkHeight(TreeNode* root) {\n"
                "    if (root == nullptr) return 0;\n"
                "    int leftHt = checkHeight(root->left);\n"
                "    if (leftHt == -1) return -1;\n"
                "    int rightHt = checkHeight(root->right);\n"
                "    if (rightHt == -1) return -1;\n"
                "    if (abs(leftHt - rightHt) > 1) return -1;\n"
                "    return max(leftHt, rightHt) + 1;\n"
                "}"
            )
        },
        {
            "question": "Find an element in a sorted array rotated an unknown number of times.",
            "ground_truth": (
                "Use a modified binary search. In a rotated sorted array, when you divide it in half, at least one of the halves "
                "will always be normally sorted. Determine which half is sorted, check if the target falls within the boundaries of that sorted half, "
                "and recurse/iterate accordingly. Handle duplicate elements by shrinking boundaries if left, right, and mid elements are identical."
            )
        },
        {
            "question": "Write a method to return all subsets of a set (Power Set).",
            "ground_truth": (
                "To compute all subsets (the power set), use recursion or combinatorics. In the recursive approach, "
                "start with an empty set of subsets. For each element in the input set, iterate through all existing subsets generated so far, "
                "clone them, and append the current element to form new subsets.\n\n"
                "Code Concept:\n"
                "vector<vector<int>> getSubsets(vector<int>& set, int index) {\n"
                "    vector<vector<int>> allsubsets;\n"
                "    if (set.size() == index) {\n"
                "        allsubsets.push_back(vector<int>());\n"
                "    } else {\n"
                "        allsubsets = getSubsets(set, index + 1);\n"
                "        int item = set[index];\n"
                "        vector<vector<int>> moreSubsets;\n"
                "        for (vector<int> subset : allsubsets) {\n"
                "            vector<int> newSubset(subset);\n"
                "            newSubset.push_back(item);\n"
                "            moreSubsets.push_back(newSubset);\n"
                "        }\n"
                "        allsubsets.insert(allsubsets.end(), moreSubsets.begin(), moreSubsets.end());\n"
                "    }\n"
                "    return allsubsets;\n"
                "}"
            )
        }
    ]