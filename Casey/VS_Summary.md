# Summary
### [*Signal Subgraph Estimation Via Vertex Screening* S. Wang et al, 2018, arXiv](https://arxiv.org/abs/1801.07683)
#### Casey Weiner 
____________________________
*The Problem*: Given a set of graphs with corresponding labels, we want to predict the label based on the graph structure.
Especially with large data sets, it is important to find a subgraph, a subset of nodes and edges, that are 
most likely responsible for the differences in graph type. However, this still proves to be a challenge since the 
number of subgraphs increases exponentially with number of vertices. With large graphs like these, feature selection
is taxing time-wise and computing-wise.  

*The Method*: The workaround for this is to use the Pearson Correlation to rank variables, as well as with MLE. 
However, Pearson's correlation only describes *linear* dependence. In order to characterize dependence in 
more general terms, a version of Dcorr called multiscale generalize correlation (MGC) is used. 
Three steps to method:
- Feature computation/extraction
- Dcorr Calculation
- Thresholding  

*Setup*: Given a set of graphs, each with a scalar label associated with them, the signal subgraph(SS) is the 
minimum number of vertices such that the graph without any of the signal subgraph edges or vertices is independent
of the label vector. Independence is defined in this context with a joint distribution. Also, it is assumed there
is a *unique* SS for every graph. The random graphs generated for testing this algorithm
used the Erdos-Renyi model of random graphs, which is optimally classified by a Bayes classifier. Thus, this
will be what is compared to the performance of the SS algorithm. 

*Feature Extraction*: This is done with two methods.  
**Non-Iterative Screening**:
The first is iterative screening, where the feature vectors 
for each vertex are either a row of the adjacency matrix or are calculated with other methods, such as 
Adjacency Spectral Embedding (ASE). This paper just uses a row from the adjacency matrix. Dcorr is calculated
between each feature vector and corresponding label vector. Then, order these correlations by their magnitudes, 
threshold them by a critical value, and the vertices that are left are the estimated SS.  
**Iterative Screening**:
The second is iterative screening, where the first method is iterated many times to eliminate "noise vertices".
The more noise vertices, the correlation goes to zero. This method finds the again the correlations of every vertex,
but now it extracts every vertex whose correlation is in some quantile of the ordered correlation vector.
Then, the previous step is repeated. This occurs until the magnitude of the vector of vertices is less
than one. Then the index of the entry in the vertex vector that outputs the maximum correlation is
found and the vector of those vertices is the SS. 

*Justification*: As noise goes to zero, the correlation continually increases. It is shown that using
SS and then classifying the data with a Bayes plugin classifier is more accurate and faster than classifying with
just Bayes. The whole graph converges more quickly to the Bayes optimal. 

*Experimentation*: Used on IER random graphs, human MRI differentiation by sex, and mouse DTI MRI differentiated
by sex. In all cases, either the iterative methods or Bayes performed most optimally.

*Conclusion*: This method, backed strongly by direct mathematical proof, is shown to be effective and accurate
at SS estimation. This method does not solely rely on linear relation, but rather general correlation that
allows for SS estimation based on things like non-scalar response or repsonse with non-Euclidean quantifiers.
This method is a widely applicable and capable tool for tackling supervised learning problems on graphs.



