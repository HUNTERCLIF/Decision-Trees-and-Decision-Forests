# Decision-Trees-and-Decision-Forests

Programming Language: Python (version 3.9.0)

Code Structure:
- decision_tree.py: Main code file.
- readme.txt: Instructions and details about the code.

Decision Tree Classifier Readme

1. Introduction:
   This program implements a decision tree classifier and random forest for binary classification problems. It uses the provided datasets or custom datasets for training and testing.

2. Files:
   - decision_tree_classifier.py: Main Python script implementing the decision tree and random forest classifier.
   - pendigits_training.txt: Training dataset for the pendigits classification task.
   - pendigits_test.txt: Test dataset for the pendigits classification task.
   - readme.txt: This file.

3. How to Run:
   - Install the Modules;
   on your terminal run: pip install numpy.
   - Update URLs or paths to local dataset files in the decision_tree_classifier.py script.
   - Run the program using the command:
   python decision_tree.py
   
4. Additional Notes:
   - The program supports two training options: "optimized" and "randomized."
   - For "randomized," the attribute is chosen randomly at each non-leaf node.
   - For "optimized," the program identifies the optimal attribute and threshold combination for each non-leaf node.
   - The output includes information about each test object, including the predicted class, true class, and accuracy.
