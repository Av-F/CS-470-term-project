# Our imports for this
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#TEAM 1: Amanda, Ryan, Avraham, Andrew

# Use pandas to read the csv file
df = pd.read_csv("natops_processed.csv")  # Ensure that the file is in the correct directory

# We will want the columns that start with fea for features
# From the pandas documentation https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
feature_cols = [col for col in df.columns if col.startswith("fea")]
# Use X as the data frame explained here https://www.geeksforgeeks.org/python-pandas-dataframe/
# Essentially we want a data frame of exclusively of features
X = df[feature_cols]

# We will need to scale the data frame for the rest of the process to work, similar to the earlier homeworks here
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Using the Elbow method we will plot the inertias of the KMeans algorithm over incrementing k# clusters starting with 1 cluster.
def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans= KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)
    fig, _ = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

optimise_k_means(X_scaled, 20)

# Use KMeans to cluster
n_clusters = 11 # We expect an optimal number of 11 clusters from our analysis using the elbow method
kmeans = KMeans(n_clusters=n_clusters, random_state=42) # Using the KMeans function, the heavy lifting is done for us
df['cluster'] = kmeans.fit_predict(X_scaled) #Save what was done in a cluster column

# We wanna count the ratios of the clusters 
# Group the SID by cluster, then Call value_counts to see how many clusters per sid
# Then convert into a wide table and if the table has no value, give the value a 0 instead of a NaN
cluster_ratios = (df.groupby('sid')['cluster'].value_counts(normalize=True).unstack(fill_value=0).reset_index())

# rename the cluster columns so that we can see each individual cluster easier
cluster_ratios.columns = ['sid'] + [f'cluster_{i}_ratio' for i in range(n_clusters)]

# Find the meta info and then merge it into the final data frame
meta_info = df.groupby('sid')[['is_test', 'class']].first().reset_index()
final_df = meta_info.merge(cluster_ratios, on='sid')

# Finally, take the data frame and make it into a csv file, then print out hte first 5 rows. 
final_df.to_csv("natops_cluster_ratios.csv", index=False)
print(final_df.head())