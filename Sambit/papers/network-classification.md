# Network Classification with Applications to Brain Connectomics

Authors: Joshua T. Vogelstein, William R. Gray, R. Jacob Vogelstein, and Carey E. Priebe

Link: <https://arxiv.org/pdf/1701.08140.pdf>

## Abstract

- purpose: Study classification of networks with labeled nodes
- existing approaches treat all edgeweights as a long vector or ignore network structure and focus on graph topology while ignoring edge weights
- propose method that uses edge weights as predictors, but incorporates network structure

## 1 - Introduction

- motivated by brain connectomics
- each unit is represented by own networks and nodes are labeled and shared across networks
- focus on functional connectivity (fMRI), specifically resting-state fMRI
  - allows to represent fMRI as single graph since time dimension can be averaged
- COBRE (54 schizo, 70 control), UMich (39 schizo, 40 control) were compared against
- parcellation is 264 ROIs deivided into 14 functional systems
- measured correlation by using marginal correlations between time series
- previous graph classification attempts
  - finds on subgraphs which is only possible computationally on small networks -> Setting tested on 20 nodes
  - define similarity measure between two networks on graph kernel -> not better than random guessing
  - reduce network to a global summary of measures like average degree, clustering coefficient, or average path length -> harm accuracy and don't identify local differences
  - classify large networks by treating edge weights as a "bag of features" -> Effectiveness depends on parcellation to define nodes, limited interatability based on edges
- network structure incorporation is better at classification and interatibility
- __GOAL OF PAPER: develop a high-dimensional network classifier that usese edge weights while respecting network structure__

## 2 - A framework for node selsection in graph classification

- Goal: Predict class <img src="https://latex.codecogs.com/svg.latex?\inline&space;Y" title="Y" /> from graph adjacency matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />
- Can easily extend for multi-class
- Standard - Construct linear classifier <img src="https://latex.codecogs.com/svg.latex?\inline&space;Y" title="Y" /> from linear combination of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />
- If <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathcal{B}&space;=&space;\left\{&space;B&space;\in&space;\mathbb{R}^{N&space;\times&space;N}:&space;B&space;=&space;B^T,&space;\mathrm{diag}(B)&space;=&space;0&space;\right\}" title="\mathcal{B} = \left\{ B \in \mathbb{R}^{N \times N}: B = B^T, \mathrm{diag}(B) = 0 \right\}" />, the loss function is

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?l(B)&space;=&space;\frac{1}{n}&space;\sum_{k=1}^n&space;\tilde{l}&space;(Y_k,&space;A^{(k)};&space;B)" title="l(B) = \frac{1}{n} \sum_{k=1}^n \tilde{l} (Y_k, A^{(k)}; B)" />
</p>

- the optimization algorithm can work with any convex and continuously differentialbe loss function
- focus on convex formulations that allow for efficient computation and small nodes
- group lasso is used to eliminate groups of variables, it is

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Omega_{\lambda,&space;\rho}&space;(B)&space;=&space;\lambda&space;\left(&space;\sum_{i=1}^N&space;||B_{(i)}&space;||_2&space;&plus;&space;\rho&space;||&space;B&space;||_1&space;\right&space;)" title="\Omega_{\lambda, \rho} (B) = \lambda \left( \sum_{i=1}^N ||B_{(i)} ||_2 + \rho || B ||_1 \right )" />
</p>
