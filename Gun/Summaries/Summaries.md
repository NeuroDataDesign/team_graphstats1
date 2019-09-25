# Summaries

## GraSPy: Graph Statistics in Python
https://arxiv.org/abs/1904.05329 arXiv, J.Chung et al.

- GraSPy was made to work with random graphs/populations of graphs, and analyze them with a statistical approach.
- Existing functions can perform preprocessing on graphs. The paper specifically gives examples of connectome model fitting in Figure 2.
- Graphs can be visualized as adjacency matrices. (Ask about
embedded graphs).


## Network Classification with Applications to Brain Connectomics
https://arxiv.org/abs/1701.08140 arXiv, J.Arroyo et al.

### Section 1:
- Networks can be created from the brain by forming weighted connections, representing correlating activity,  between nodes, representing brain regions. This paper in particular focuses on functional connectivity, as indicated by correlations in BOLD signal.
- Connectivity is measured for every pair of nodes, forming an nxn adjacency matrix. For the datasets used,  their sizes make some approaches non-viable. As a result, one may approach analysis by first reducing the network to certain global features, but will ignore local differences as a result.
- The other approach focuses on local features, but forgoes the analysis of global structure.

### Section 2:
- Consider a graph with N nodes such that for adjacency matrix A∈R<sub>nxn</sub>, A<sub>ij</sub> = A<sub>ji</sub> and A<sub>ii</sub> =0. Aim to classify
the network in a binary manner.

**2.1**
- For a penalized classification approach, create a linear combination <A,B> = Tr(B<sup>T</sup> A) where B is found through an optimization problem with penalties to emphasize network structure. This problem is solved with an argument B such that the sum of a logistic loss function and a penalty is minimized with respect to B.
- The goal is to predict the response Y from <A,B>.
- The sparsity penalty is formed to promote the minimal vertex cover problem for a group of selected edges. Other penalties can be formed to enforce spatial properties.

**2.2**
- The group lasso approach is designed to minimize the number of nodes and edges through sparsity penalties
- This penalty is defined as a multiple of the sum of the l2 norms for each of the rows or columns of B added to a multiple of the l1 norm of B. The l1 norm allows for selection of subsets of edges connected to active nodes.

### Section 3:
- Proximal algorithm is applied to the main problem, with ADMM solving for the proximal operator at each step.
- ADMM breaks down convex optimization problems into smaller pieces that are easier to solve.

### Section 4:
- Actual math theory, come back later

## Decision Forests for Classification, Regression, Density Estimation, Manifold Learning and Semi-Supervised Learning
https://www.microsoft.com/en-us/research/publication/decision-forests-for-classification-regression-density-estimation-manifold-learning-and-semi-supervised-learning/ Microsoft TR, A.Criminisi et al.

### Section 1:
- Using ensemble methods gives better accuracy, hence random forests instead of trees
- Decision trees are randomly trained. Bagging randomly samples labelled training data

## Matched Filters for Noisy Induced Subgraph Detection
https://arxiv.org/pdf/1803.02423.pdf D.Sussmah et al.

### Introduction
- The matched filter allows for comparisons between graph representations with sets of vertices of different cardinality.
- Graph matching matched filter: Aligning smaller networks to the closest subgraph in the larger network.

### Background for Graph Matching
**2.1**
- Noisy matched filters: Search for the closest induced subgraph.
- Try to determine when it's possible to narrow down to a specific vertex correspondence.

**2.2**
- There are approaches guaranteed to find an approximate global minimum, but they scale poorly.
- Relaxation of constraints coupled with random restarts to avoid poor local minima.
- Seeded vertices increase efficiency and accuracy.

**2.3**
- corrER: correlated Erdos-Renyi
- corrER model: Two ER random graphs that share a common vertex set, with a correlation between the adjacencies of two vertices in each of the graphs.
- http://jmlr.org/papers/volume15/lyzinski14a/lyzinski14a.pdf for more info on corrER
- Should probably come back to this section for the algorithm

### Padding Approaches
- Match two graphs with different vertex cardinality by padding the smaller graph with rows/cols.
- **Naive Padding:** Add extra rows and columns to the bottom and right of the matrix.
- **Centered Padding** and **Oracle Padding** - Come back later because I have no idea why this works 
