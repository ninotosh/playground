from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

n_clusters = 2
documents = [
    'The quick brown fox jumps',
    'over the lazy dog',
    'The quick brown cat leaps',
    'over the lazy bird',
]

vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
vector = vectorizer.fit_transform(documents)
print("documents: %d, unique words: %d" % vector.shape)
# documents: 4, unique words: 9

kmeans = MiniBatchKMeans(n_clusters=n_clusters, verbose=False)
kmeans.fit(vector)

for cluster_idx, document in zip(kmeans.labels_, documents):
    print(cluster_idx, document)
# (0, 'The quick brown fox jumps')
# (1, 'over the lazy dog')
# (0, 'The quick brown cat leaps')
# (1, 'over the lazy bird')

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for cluster_idx in range(n_clusters):
    print("Cluster %d:" % cluster_idx),
    for i in order_centroids[cluster_idx, :2]:
        print('%s' % terms[i]),
    print
# Cluster 0: quick brown
# Cluster 1: lazy dog

print kmeans.partial_fit(vectorizer.fit_transform(['cat jumps', 'lazy bird over the'])).labels_
# [0 1]
