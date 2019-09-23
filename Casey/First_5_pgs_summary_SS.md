# Summary
### [*Graph Classification using Signal-Subgraphs: Applications in Statistical Connectomics* J. Vogelstein et al, 2012, arXiv](https://arxiv.org/abs/1701.08140)
#### Casey Weiner  
##### Page 1:
> *Labeled graph classification*: Can we correctly infer the class for a new graph given a collection of known graphs? 
This assumes each node has a unique label and that all graphs have the same number of vertices with the same labels. 
This model is built on the idea of a class-dependent signal represented by a subset of edges, called the *signal subgraph* (SS). 
The signal subgraph amounts to the differences between the graph classes being compared. Interested in *coherance*,
or the extent to which the SS edges are incident to a comparably small number of vertices.
This introduces the idea of *sparsity*, defined later. Analysis using only vertex labeling accounts for
the labels of each node but neglects overall graph structure. Vice-verse for analyis using only graph-theory parameters,
including *weighted degree* or *clustering-coefficient*. Know not only the graph structure of an afflicted
brain, but the *specific vertices or edges* that are altered.  
Need to utilize
> - *Unique* Vertex Labels 
> - Graph Structure  

##### Page 2:
> Methodology used on magnetic resonance connectomes differentiated by sex. Utilizing graph structure
can significantly enhance classification accuracy. However, the best strategy for a data set depends on
the size and model and modality of the data. The goal is to build a graph classifier that can take in a new 
graph g and output its class y. The graph classifiers need to be *interpretable* with respect to the vertices
and edges of the graph. Adhere to practice of identifying graphs with their adjacency matrices. Any variability
between graphs is in the adjacency matrix. All have the same set of distinctly labeled vertices. Edges are considered
independent. Assume the graphs are *simple* graphs:
> - Undirected  
> - Binary Edges  
> - Lack Self-Loops  

> The likelihood of an edge between two vertices is given by a Bernoulli random variable. The likelihood parameters
for all edges that are not signal-edges have no class-dependent signal. We utilize essentially a Boolean 
loss function.

##### Page 3:
> *Risk*, on the other hand, is the expected loss under the true distribution. The best classifier for a 
distribution minimizes risk: this is called the *Bayes optimal classifier*. These, however, are not often not
possible for the data, so it is neccessary to create an estimate for a classifier from *training data*. Therefore,
it is possible to establish a *Bayes plugin classifier* for graphs. S, pi, and p must be provided with estimators.
Estimators must follow the following criteron:
> - **Consistent**: Converges to the actual value as the number of samples goes to infinity.
> - **Robust**: Relatively uninfluenced by small model misspecifications.
> - **Quadratic Complexity**: Computational time complexity is no more than a quadratic function of the 
number of vertices.
> - **Interpretable**: The parameters are interpretable with respect to a subset of vertices/edges.  

> It is impossible to just test every possible subgraph: it scales exponentially with base two. Each edge
can be evaluated seperately, although it isn't necessarily the right choice to make every edge independent.
The null hypothesis is that the distributions for the edges are the same, ie the edges come from graphs of 
the same class type. This is tested with hypothesis testing through test statistics. Null is rejected whenever
the value of the test statisic surpasses some critical value. 

##### Page 4:
> It is not possible (currently) to search for every possible subgraph composed of V vertices and s edges.
We want to minimize the number of edges that need to be rejected under the null. In other words, minimize the number of non-differentiating edges.
An estimate of the signal-subgraph is the collection of s edges with minimal test statistics. The *incoherent signal-subgraph estimator*
is given by the set of s edges ordered by their test statistic significance. The number of unique test statistic
values is actually often much less than the maximum possible number. As a result many pairs of edges will be "ties",
where each one is equally valid. Another assumption is that all of the edges in the SS are incident to 
one of *m* nodes called *signal vertices*, which are just vertices in the SS. The process for finding the set 
of edges with minimal test statistics is to first compute the significance of each edge. Then rank each subset of
edge, starting from one node to every other, by significance. Then initialize critical value at zero and then
assign each vertex a score equal to the number of edges that, when incident to that vertex, are more significant
than the critical value. Then sort those scores. Then check if there are m vertices whose scores sum to greater
than the size of the SS. If that is true, the minimization problem is solved and then the collection of s most 
significant edges is the *coherent signal-subgraph estimate*. If that is false, increment c and go back to the fourth step.
Ties are broken arbitrarily.

##### Page 5:
> Coherent signal-subgraphs allow for the creation of "coherograms", which are plots where the number of rows
is equal to the number of nodes and the number of columns is equal to the number of different critical values.
The elements of the matrix are the scores for each node at each critical value. This provides a visual description
of the coherence of the SS. The likelihood parameter for an edge between nodes u and v is a Bernoulli parameter
for each edge in the class. The maximum likelihood estimator (MLE) is the average value of each edge per class and is 
used to estimate the parameter.
