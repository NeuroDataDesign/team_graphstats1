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

- since <img src="https://latex.codecogs.com/svg.latex?\inline&space;B" title="B" /> is symmetric, it makes the groups overlap, and the last terms promotes sparsity
- Can optimize over the adjacency matrix, which has advantages since it does not require lass penalty
- approach similar to selecting signal nodes, but relies on two different coefficients for the same edge <img src="https://latex.codecogs.com/svg.latex?\inline&space;B_ij,&space;B_ji" title="B_ij, B_ji" />
- can also assign these edges to the same group, simplifying the penalty (can also extend to matrix of coefficients)
- combining ridge and lasso penalties can be beneficial as an additional tuning parameter
- the total graph classifier requires solving the following problem

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?(\hat{B},&space;\hat{b})&space;=&space;\arg&space;\min_{B&space;\in&space;\mathcal{B},&space;b&space;\in&space;\mathbb{R}}&space;\left\{&space;\frac{1}{n}&space;\sum_{k=1}^n&space;\log&space;\left(1&space;&plus;&space;\exp&space;(-Y_k&space;\langle&space;B,&space;A^{(k)}&space;\rangle&space;&plus;&space;b)&space;\right)&space;&plus;&space;\frac{\gamma}{2}&space;\|&space;B&space;\|^2_F&space;&plus;&space;\lambda&space;\left(&space;\sum_{i=1}^N&space;\|B_i\|_2&space;&plus;&space;\rho&space;\|&space;B&space;\|_1&space;\right&space;)&space;\right\}" title="(\hat{B}, \hat{b}) = \arg \min_{B \in \mathcal{B}, b \in \mathbb{R}} \left\{ \frac{1}{n} \sum_{k=1}^n \log \left(1 + \exp (-Y_k \langle B, A^{(k)} \rangle + b) \right) + \frac{\gamma}{2} \| B \|^2_F + \lambda \left( \sum_{i=1}^N \|B_i\|_2 + \rho \| B \|_1 \right ) \right\}" />
</p>

## 3 - The optimization algorithm

- uses proximal algorithms and alternating direction method of multipliers (ADMM)
- main difficulty comes from overlapping groups
  - can use subgradiient descent (slow rate of convergence)
  - proximal based on smoothing (sparsity patterns may not be preserved)
- ADMM gives more accurate sparsity and flexibility in algorithm
- can optimize extra paramater <img src="https://latex.codecogs.com/svg.latex?\inline&space;b" title="b" /> in logistic loss function by using Newton's method
- can formulate as optimization problem now and use a Lagrangian to solve
- can easily introduce new panalties by rewriting a paramater <img src="https://latex.codecogs.com/svg.latex?\inline&space;\tilde{Q}" title="\tilde{Q}" /> to

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{1}{2}&space;\|&space;\tilde{B}&space;-&space;Z^{(k)}&space;\|^2_2&space;&plus;&space;t&space;\lambda&space;\left(&space;\sum_{i=1}^N&space;\|&space;Q_{(i)}&space;\|_2&space;&plus;&space;\rho&space;\|&space;R&space;\|_1&space;\right)&space;&plus;&space;t&space;\Psi(\tilde{Q})" title="\frac{1}{2} \| \tilde{B} - Z^{(k)} \|^2_2 + t \lambda \left( \sum_{i=1}^N \| Q_{(i)} \|_2 + \rho \| R \|_1 \right) + t \Psi(\tilde{Q})" />
</p>

- using the signal subgraph way by optmizing over <img src="https://latex.codecogs.com/svg.latex?\inline&space;B" title="B" /> simplifies the penalties

## 4 - Theory

- the panalized problem can recover correct subgraph
- want to focus whether of subset of active notes is correctly estimated via the subset of nodes
- assume that the loss function is convex and there are bounds to size of loss Hessian
- these can be substititued for bounds in probability in the case of random designs; requires distribution on first derivative
- at least one edge to all active nodes needs non-zero weight to recover all active nodes

## 5 - Numerical results on simulated networks

- evaluate performance on synthetic networks
- use SBM because community structure, use Gaussian since fMRI
- fig 3 offers good way to benchmark my performance, fig 5 for classification accuracy, fig 6 for cross vallidation, fig7 for real data
