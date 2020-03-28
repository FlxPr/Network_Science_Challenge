# Feature Engineering
## Ensemble Learning Pipeline
### Features that we use
#### Quick Summary
        'similarity_A_B', 'similarity_Aout_B', 'similarity_A_Bin', 'aut_common','n_previously_cited', 'reversed', 'adar', 'jaccard', 'jaccard_weak','adar_weak', 'ID_X1', 'Year_X1', 'betweenness_centrality_X1','eigenvector_centrality_X1', 'in_closeness_centrality_X1','in_degree_centrality_X1', 'out_closeness_centrality_X1','out_degree_centrality_X1', 'pagerank_X1', 'infomap_X1', 'ID_X2','Year_X2', 'betweenness_centrality_X2', 'eigenvector_centrality_X2','in_closeness_centrality_X2', 'in_degree_centrality_X2','out_closeness_centrality_X2', 'out_degree_centrality_X2','pagerank_X2', 'infomap_X2'
#### Node Features
##### Centralities
        Here the intuition is simple. Nodes that have high out-degree centralities are more likely to cite other papers (they must have done an extensive research for their work), while nodes that have high in-degree centralities are likely to be cited (eg. they may have made some breakthrough in their research field). 

        On the other hand, betweenness, closeness, eigenvector and pagerank centrality capture how important a node is in a graph, which we use a proxy to the importance of a paper.

        Given those basic node features, along with node IDs, our baseline Xgboost model achieved an F1 score of about 90%.
##### Doc2Vec
        As abstracts and titles provide us abundant information regarding the content of each paper, we can expect that one paper is more likely to cite another if they are similar in content and field of research. In other words, if we can somehow determine that paper A is about physics, while paper B is related to psychology, then A is unlikely to cite B even if B has a high importance socre in the graph.

        Therefore, for each node, we concatenate the title and abstract together, and convert them to a 50-element vector using the Doc2Vec algorithm.

        However, if we just use the doc vector as a feature for our classifier, we will end up with over 100 features, which significantly slows down the training process and does do much good in terms of performance. Therefore, we do not feed those raw vectors into our final model. Instead, we calculate some similarity metrics as can be seen in edge feature section below.
##### Community Detection
        Community detection is a technique by which you can cluster nodes in a graph to sub communities. Here we assume papers in the same field of research form different sub groups and they cite each other more often than they cite papers outside their groups. As a fictional example, economists from the neo-classical school and Austrian school have differnt views towards the economy and therefore may be at odds with each other, let alone cite the counterpart's work.

        Therefore, we perform the infomap clustering algorithm, which is sclable for large graphs, and assign each node to a community number, slightly boosting our F1 score to about 92%. 
#### Edge Features
        Here let us assume that our task is to determine whether paper A has cited paper B.
##### Doc Similarity
        Continuing on the previous Doc2Vec section, here we define 3 similarity measures:
        1. The similarity between A and B.
        2. The similarity between B and those cited by A
        3. The similarity between A and those citing B

        All of the metrics above are calculated by using the cosine similarty between the average doc vectors. Take the second metric for exmaple; if A cites C and D, then we use the cosine simialrity between B and the mean of the doc vector of C and D. 

        Here we are trying to account for the case where, for example, B is related to psychology but is frequently cited by behavior economists, and then if A looks like a paper in behavior economics, then A has a high chance of using B's work. This set of features takes our F1 score to a high of 96%.
##### Author Reference Frequency
        Author reference frequency refers to the number of times authors of A has cited the work of the authors of B before. Intuitively, if authors of A are particularily fond of or interested in some scholars, they are likely to cite their work.

        Althought theoretically justified, this feature has a drawback: the author column is a mess. The same author could have 5 different strings due to spelling errors, whether the first name is an initial or full spelled (A. vs Anna), and the presence of middle names and school names. Therefore, we design a cleaning pipeline to take only the initial and last name of each author, significantly reduing the noise and inaccuracy present in our dataset.
##### Reversed Citation
        Since citation happens chronologically, if we know for sure that A has cited B from the training set, it is impossible for B to cite A in the test set!
##### Adamic/Adar Index and Jaccard Similarity
        The last set of features accounts for the similarity of A and B using their common neighbors. We can expect A to cite B if they are similar in terms of graph structure, and it is these 3 final sets of features that bring us an F1 score of 99%! 
### Features we created but discarded later
#### Shortest Path
        We attemp to incorporate the shortest path of A and B into our feature set. However, unfortunately it results in a data leakage during the Cross Validation since the shortest distance will be 1 if A has cited B! To mitigate that problem, we could still take the shortest distance but only after the removal of the edge between A and B if it exists. However, it is too computationally expensive to implement using the IGraph package. Therefore, we opt for other easier-to-compute features. 
#### Author Importance Score
        We also create an author graph where each node represent an author and each edge represents a citation from our dataset. And then by computing the pagerank of each author we can obtain an author importance score and use it as a node feature (if a paper is co-authored by several people, we take the maximum importance socre). However, it does not improve our score. The centrality measure above has well gave us a sense of how important each node is!
#### Node2Vec
        The last technique we have tried is Node2Vec, by which you can obtain a vector representation of each node from a graph. However, we end up with having an addtional 100 columns due to each 50-element node vector, which gives a huge burden to our Xgboost classifier and slightly takes down our F1 score. It is better to feed our model well-defined features such as centralities rather than black box vectors generated. 

## Graph Convolutional Network Pipeline
### Features that we use
        For this model, we use only year of publications and Doc2Vec vectors as features for our neural network, as we expect our model to capture all the graph-realted information such as centralities and edge&node similarities on its own.

# Modeling tuning and comparison
## Ensemble Learning Pipeline
### Xgboost
#### Hyperparameter Tuning and Performance
        Before embarking on feature engineering, we did a quick submission using only node IDs and achieved an F1 score of about 76% as a baseline.

        We have tried multiple combinations of estimator numbers and learning rates manually and the following parameters stand out:
        Parameters: 500 estimators and 0.3 learning rate
        CV Train F1: 99.99%
        CV Test F1: 99.92%
        Leaderboard F1: 99.84%

        Since there is almost no overfitting, we do not try to tune any hyperparameters to reduce the complexity of the trees.

### Extra Trees
#### Hyperparameter Tuning and Performance
        From our experiment, we decide that after 300 there is almost no gain in increasing the number of estimators. The following is its performance:
        CV Train F1: 100%
        CV Test F1: 99.74%

        Here it achieves a similar but a bit worse result than Xgboost. Nonetheless, we perform a grid search on the maximum depth and maximum features in the hope to address its overfitting problem. The result suggests a maximum feature of 8. However, it merely raises the test CV score by 0.01%.

### Logistic Regression
#### Performance
        As another baseline, a plain vanilla logistic regression is also employed. The following is the performace:
        CV Train F1: 79%
        CV Test F1: 79%
        Even though with the all the hand-engineered features, it performs almost as bad as the baseline Xgboost model with only index features.        


## Graph Convolutional Network Pipeline
### Background
#### Graph Convolutional Network
xxx-(Add some nice pictures and copy some technical details)
#### Attetion Mechanism
xxx-(Add some nice pictures and copy some technical details)
### Architecture
    First part: node embedding layers
    Second part: get edge embeddings by concatenating the node embeddings of two nodes together
    Third part: fully connected layers for classification
### Hyperparameter Tuning and Performance
    We try to modify the number of hidden convolutional layers and also the number of hidden neurons inside each convolutional block, but it merely makes a difference. No matter how hard we try, it can never achieve a result as good as our ensemble models do.

    GCN modules Train F1: about 76%
    GCN modules Test F1: about 75%
    Leaderboard F1: about 75%
    GAT modules Train F1: 92%
    GAT modules Test F1: 91%
    Leaderboard F1: about 91%





Left to do:
# Find some papers to justify my feature engineering
(e.g I used doc2vec, so we could cite papers boasting about its performance or papers of its inventors)
# Add some technical details to features such as adar index
# EDA