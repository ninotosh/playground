[unsupervised_doc_clustering.py](unsupervised_doc_clustering.py) is a demo python script of incremental document clustering.

First install scikit-learn

```
pip install -U numpy scipy scikit-learn
```

Then run

```
python main.py
```

The output will be like

```
documents: 4, unique words: 9
(0, 'The quick brown fox jumps')
(1, 'over the lazy dog')
(0, 'The quick brown cat leaps')
(1, 'over the lazy bird')
Cluster 0: quick brown
Cluster 1: lazy bird
[0 1]
```

4 documents are grouped into 2 clusters.
Cluster 0 has

```
(0, 'The quick brown fox jumps')
(0, 'The quick brown cat leaps')
```

and cluster 1 has

```
(1, 'over the lazy dog')
(1, 'over the lazy bird')
```

Typical words found in each cluster are

```
Cluster 0: quick brown
Cluster 1: lazy bird
```

These 2 documents are additionally clustered:

```
cat jumps
```

```
lazy bird over the
```

The last line of the output

```
[0 1]
```

indicates the first additional document belongs to cluster 0, the second cluster 1.
