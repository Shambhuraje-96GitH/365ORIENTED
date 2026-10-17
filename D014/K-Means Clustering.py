# Day 14 - K-Means Clustering
# 365-Day AIML Engineer Challenge

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# --------------------------------------------------
# 1. Create sample customer data
# --------------------------------------------------

data = np.array([
    [22, 15],
    [25, 18],
    [28, 20],
    [30, 22],
    [35, 25],
    [40, 30],
    [45, 35],
    [50, 40],
    [55, 45],
    [60, 50],
    [65, 55],
    [70, 60]
])


# --------------------------------------------------
# 2. Apply K-Means Clustering
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(data)


# --------------------------------------------------
# 3. Get cluster labels
# --------------------------------------------------

labels = kmeans.labels_

print("Cluster Labels:")
print(labels)


# --------------------------------------------------
# 4. Get cluster centers
# --------------------------------------------------

centers = kmeans.cluster_centers_

print("\nCluster Centers:")
print(centers)


# --------------------------------------------------
# 5. Predict cluster for a new customer
# --------------------------------------------------

new_customer = np.array([[42, 32]])

prediction = kmeans.predict(new_customer)

print("\nNew Customer:")
print(new_customer)

print("Predicted Cluster:")
print(prediction[0])


# --------------------------------------------------
# 6. Visualize the clusters
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data[:, 0],
    data[:, 1],
    c=labels,
    s=100
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    label="Cluster Centers"
)

plt.xlabel("Age")
plt.ylabel("Annual Spending Score")

plt.title("K-Means Customer Clustering")

plt.legend()

plt.show()


# --------------------------------------------------
# 7. Elbow Method
# --------------------------------------------------

inertia_values = []

for k in range(1, 8):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(data)

    inertia_values.append(model.inertia__)


# --------------------------------------------------
# 8. Plot Elbow Curve
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 8),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.title("Elbow Method for Optimal K")

plt.xticks(range(1, 8))

plt.show()


# --------------------------------------------------
# End of Day 14
# --------------------------------------------------

print("\nDay 14 K-Means Clustering completed successfully!")