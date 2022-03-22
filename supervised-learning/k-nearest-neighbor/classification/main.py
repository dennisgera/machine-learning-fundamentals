import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

## load breast cancer dataset
breast_cancer_data = load_breast_cancer()

## investigate data
print (breast_cancer_data.data[0])
print (breast_cancer_data.feature_names)
print (breast_cancer_data.target)
print (breast_cancer_data.target_names)

## split the data into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = .2, random_state = 0)

# confirm training and validation lengths
print (len(training_data), len(training_labels))

## run the k nearest neighbors classifier
k_list = []
accuracies = []
for k in range(1,101):
  # create classifier with k nearest neighbors
  classifier = KNeighborsClassifier(n_neighbors = k)
  # fit the classifier to the training data
  classifier.fit(training_data, training_labels)
  # append k value to k_list
  k_list.append(k)
  # get classifier score on validation data/label and append to accuracies
  accuracies.append(classifier.score(validation_data, validation_labels))

## create a single list of k, accuracies tuples
combined = list(zip(k_list, accuracies))
print (combined)

## graph the results
plt.plot(k_list, accuracies)
plt.xlabel('k-neighbors')
plt.ylabel('Accuracies')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()


## find the k value that generated a model with greatest accuracy
max_acc = 0
k_max = 0
for point in combined:
  if point[1] > max_acc:
    max_acc = point[1]
    k_max = point[0]
print (k_max, max_acc)    
