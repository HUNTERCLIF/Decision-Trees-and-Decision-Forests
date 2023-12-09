import urllib.request
from io import StringIO
import random
import math
import numpy as np
from collections import Counter


class Node:
    def __init__(self, attribute=None, threshold=None, value=None, right=None, left=None):
        self.attribute= attribute
        self.threshold= threshold
        self.value= value
        self.left= left
        self.right= right

def read_data_from_url(url):

    """
    Read the data from a URL and return a list of kists.
    Each innner list represents a row in the file.
    """
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    file_like_object = StringIO(data)

    parsed_data = []
    for line in file_like_object:
        row = line.strip().split(',')
        row = [float(value) if value.replace('.', '', 1).isdigit() else value for value in row]
        parsed_data.append(row)
    return parsed_data 


def calculate_entropy(labels):
    unique_labels, lables_counts = np.unique(labels, return_counts=True)
    probabilities = lables_counts /len(labels)
    probabilities[probabilities == 0] =1
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(left_subset, right_subset):
    total_entropy = calculate_entropy(left_subset + right_subset)

    # Calculate entropy after the split
    entropy_after_split = (
        (len(left_subset) / len(left_subset + right_subset)) * calculate_entropy(left_subset) +
        (len(right_subset) / len(left_subset + right_subset)) * calculate_entropy(right_subset)
    )

    # Calculating information gain
    information_gain = total_entropy - entropy_after_split
    return information_gain


def find_best_split(data):
    num_attributes = len(data[0]) - 1  # Exclude the last column (class label)
    best_attribute = None
    best_threshold = None
    max_information_gain = -1
    
    for attribute in range(num_attributes):
        values = sorted(list(set(row[attribute] for row in data)))
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

        for threshold in thresholds:
            left_subset = [row[-1] for row in data if row[attribute] <= threshold]
            right_subset = [row[-1] for row in data if row[attribute] > threshold]

            current_information_gain = calculate_information_gain(left_subset, right_subset)

            if current_information_gain > max_information_gain:
                max_information_gain = current_information_gain
                best_attribute, best_threshold = attribute, threshold

    return best_attribute, best_threshold


def construct_optimized_tree(data):
    if not data:
        return None
    
    best_attribute, best_threshold = find_best_split(data)
    node = Node(attribute=best_attribute, threshold=best_threshold)

    left_data = [row for row in data if row[best_attribute] <= best_threshold]
    right_data = [row for row in data if row[best_attribute] > best_threshold]

    node.left = construct_optimized_tree(left_data)
    node.right = construct_optimized_tree(right_data)

    return node

def construct_randomized_tree(data):
    if not data:
        return None

    # Randomly choose an attribute
    random_attribute = random.choice(range(len(data[0]) - 1))
    values = sorted(set(row[random_attribute] for row in data))
    random_threshold = random.choice([(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)])

    node = Node(attribute=random_attribute, threshold=random_threshold)

    left_data = [row for row in data if row[random_attribute] <= random_threshold]
    right_data = [row for row in data if row[random_attribute] > random_threshold]

    node.left = construct_optimized_tree(left_data)
    node.right = construct_optimized_tree(right_data)

    return node


def construct_decision_tree(data, option):
    if option == "optimized":
        return construct_optimized_tree(data)
    elif option == "randomized":
        return construct_randomized_tree(data)
    else:
        raise ValueError("Invalid option for Construct Decision Tree")
    

def classify_object(node, test_object):
    if node.value is not None:
        return node.value
    else:
        if test_object[node.attribute] <=node.threshold:
            return classify_object(node.left, test_object)
        else:
            return classify_object(node.right, test_object)


def classify_with_decision_tree(decision_tree, test_data):
        predictions = [classify_object(decision_tree, test_object) for test_object in test_data]
        return predictions

def classify_with_forest(forest, test_data):
        forest_predictions = [classify_with_decision_tree(decision_tree, test_data) for decision_tree in forest]
        aggregated_predictions = [Counter(predictions).most_common(1)[0][0] for predictions in zip(*forest_predictions)]
        return aggregated_predictions

def majority_vote(predictions):
    vote_counts = Counter(predictions)
    majority_class = vote_counts.most_common(1)[0][0]
    return majority_class


def print_output(object_index, predicted_class, true_class, accuracy):
        print(f"Object Index = {object_index}, Result = {predicted_class}, True Class = {true_class}, Accuracy = {accuracy}")


def calculate_accuracy(predicted_class, true_class):
    if predicted_class == true_class:
        return 1.0
    else:
        return 0.0
    

def print_classification_accuracy(accuracy_values):
    average_accuracy = sum(accuracy_values) / len(accuracy_values)
    print(f"Classification Accuracy = {average_accuracy}")
    

def main():
  #provide the urls for apendigits datasets.
  pendigits_training_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra'
  pendigits_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes'

  # dirrect from the urls
  pendigits_training_data = read_data_from_url(pendigits_training_url)
  pendigits_test_data = read_data_from_url(pendigits_test_url)

  print("Pendigits Training Data:")
  print(pendigits_training_data[:5])
  print("\nPendigits Test DATA:")
  print(pendigits_test_data[:5])

  forest_predictions = [[1, 2, 1], [2, 2, 2], [1, 1, 1]]  
  true_labels = [1, 2, 1]

  accuracy_values = []
  for i, (true_class, predictions) in enumerate(zip(true_labels, forest_predictions)):
        # Aggregate results from multiple trees using majority voting
        predicted_class = majority_vote(predictions)

        # Calculateing accuracy for the test object
        accuracy = calculate_accuracy(predicted_class, true_class)
        accuracy_values.append(accuracy)

        # output for the test object
        print_output(i, predicted_class, true_class, accuracy)

    # the overall classification accuracy
        print_classification_accuracy(accuracy_values)

if __name__=="__main__":
   main()









 