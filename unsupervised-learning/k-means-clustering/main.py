import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# get digits dataset from sklearn.datasets
digits = datasets.load_digits()

# get description of dataset 
print (digits.DESCR)

# find out what data and target looks like 
print (digits.data)
print (digits.target)

# visualize image 
plt.gray() 
plt.matshow(digits.images[100]) # looks like a 4
plt.show()
plt.clf()
# confirm 100's label
print(digits.target[100]) # 4

# investigate 64 sample images
# Figure size (width, height) 
fig = plt.figure(figsize=(6, 6)) 
# Adjust the subplots  
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) 

# For each of the 64 images 
for i in range(64): 
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position 
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[]) 
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest') 
    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()

# find best number of k clusters
num_clusters = list(range(1, 20))
inertia = list()
for k in num_clusters:
  model = KMeans(n_clusters=k)
  model.fit(digits.data)
  inertia.append(model.inertia_)

plt.plot(num_clusters, inertia, '-o')
plt.title('Optimized Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
plt.clf()

# optimization method did not provide valuable insight - best k number could not be identified. Since we know there are 10 digits, we will proceed with k=10

k = 10
model = KMeans(n_clusters=k)
model.fit(digits.data)

# create figure to visualize centroids
fig = plt.figure(figsize=(8,3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(k):
  # initialize subplots in a grid of 2x5, at the i+1th position
  ax = fig.add_subplot(2, 5, i+1)
  # display images
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)
plt.show()

# test my handwriting
new_samples = np.array([[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.45,1.53,0.15,0.00,0.00,0.00,0.00,1.37,7.62,7.62,4.43,0.00,0.00,0.00,0.00,0.00,1.45,5.88,6.10,0.00,0.00,0.00,0.00,0.00,1.07,7.32,4.73,0.00,0.00,0.00,0.00,0.00,4.35,7.62,5.34,4.58,5.11,0.30,0.00,0.00,3.36,7.29,7.08,6.40,5.87,0.38,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.98,1.60,0.69,0.00,0.00,0.00,0.00,0.00,7.02,7.63,7.62,6.02,0.76,0.00,0.00,0.00,7.62,4.86,3.59,7.47,4.65,0.00,0.00,0.00,7.63,3.43,0.00,4.42,7.55,0.92,0.00,0.00,6.33,7.32,4.81,5.19,7.62,1.45,0.00,0.00,0.92,4.80,7.32,7.40,3.97,0.00,0.00,0.00,0.00,0.00,0.21,0.11,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.08,2.21,2.29,1.60,0.00,0.00,0.00,0.00,1.30,7.62,7.62,6.86,0.00,0.00,0.00,0.00,0.00,0.95,6.64,5.80,0.00,0.00,0.00,0.00,0.00,1.60,7.62,3.36,0.61,0.15,0.00,0.00,0.00,5.19,7.62,7.62,7.62,3.59,0.00,0.00,0.00,1.14,3.03,3.58,3.19,0.69,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.15,1.68,1.83,0.00,0.00,0.08,4.96,6.41,7.32,7.62,7.40,0.00,0.00,3.74,7.62,7.27,4.66,2.59,0.74,0.00,0.00,3.82,6.90,7.62,4.35,0.00,0.00,0.00,0.00,0.15,4.05,7.32,5.19,0.00,0.00,0.00,0.00,0.46,6.79,5.72,1.43,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(3, end='')
  elif new_labels[i] == 1:
    print(0, end='')
  elif new_labels[i] == 2:
    print(8, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(9, end='')
  elif new_labels[i] == 5:
    print(2, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(7, end='')
  elif new_labels[i] == 8:
    print(6, end='')
  elif new_labels[i] == 9:
    print(5, end='')

  