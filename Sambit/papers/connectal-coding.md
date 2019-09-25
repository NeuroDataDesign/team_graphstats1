# Connectal coding: discovering the structures linking cognitive phenotypes to individual histories

Authors: Joshua T. Vogelstein, Eric W Bridgeford, Benjamin D Pedigo, Jaweon Chung, Keith Levin, Brett Mensh, and Carey E Priebe

Link: <https://www.sciencedirect.com/science/article/abs/pii/S0959438818301430>

## Introduction

- models of brain activity (neural coding) link patterns of brain activity to past and future events
- outline the importance of connectal codiing

## Modeling brains as networks

- "neuron doctrine" - brain is a network, and can be "Turing-complete" (solve any computational problem)
- __connectome: is an abstract mathematical model of brain structure and is a set of two kinds of objects:__
  - vertices (nodes): define spatial constraints (left/right mushroom body, whoole brain, etc.), spatial resolution (cell, cellular compartment, etc.), type (neural, glial, etc.), and developmental stage (postnatal day six, etc.)
  - edges (links): represents presence (or lack represents absence) of a connection or  communication between nodes. Constrained by kind of communication, temporal duration
- adjacency matrices specify this graph
- connectomes require additional structure such as edge weights (continuous magnitude)
- nodes have attributes
- how to measure?
  - dMRI: has both false positives and negatives
  - fMRI: same problem
  - correlations: susceptibility to exogenous variables

## Example of estimated connectomes

- C elegans (worm): only animal with complete connectome
  - every neuron is a node, every edge estimated by chemical synapses or gap junctions
  - edge estimated by total volume of synapses
  - C elegans has two sexes, male and hermaphrodite
- Drosophila (fly): larval connectome
  - left and right mushroom bodies using only chemical syapses
  - edges weighted between pair of neurons and directed
- Mus musculus (mouse): high resolution dMRI
  - network is undirected and weights are number of tracts between regions
  - we rescale between 0 and 1
- Homo sapiens (human): fMRI, anatomical, or dMRI for individuals
  - functional connectomes are Pearson correlation matrices converted to ranks
  - diffusion connectomes are normalized as described above

## The purpose of brain codes

- representation of information in the brain
- connectal codes a storage of info
- genome enccodes blueprint

## The role of connectomes in connectal coding

- connectotype is collection of nodes and edges (and maybe attributes)
- no one-to-one between connectotypes and phenontypes
- connectomes are interesting because they help understand relationship between brain structure and individual histories of cognitive phenotypes

## Models of connectomes

- most common way to model is "bag of edges" where each edge is independent
  - requires many statistical tests
  - network-based statistics lack theoretical guarantees for controlling false positives
- "bag of features" calculates multiple graph-wise or node-wise statistics
  - vastly different networks can produce the same value
  - for a connectome with <img src="https://latex.codecogs.com/svg.latex?\inline&space;n" title="n" /> there are <img src="https://latex.codecogs.com/svg.latex?\inline&space;2^{n^2}" title="2^{n^2}" /> subgraphs
  - different features are not independent of on another
- statistical modeling of networks
  - network is complex high dimensional random variagle with built-in structure or relationships
  - ignore unique node labels
  - built connectal coding on this foundation

## Statistical models of connectomes

- ER -> each edge is smpled identically and independently (Poisson)
- SBM -> each node is in a group or community and the probability of edge sampling is determined by the group
- each node as a own group ("latent position") operates on single unweighted networks with no attributes
- non-parametric Bayesians models of populations of networks have also been proposed

## Statistical model for connectal coding

- have four random variables corresponding to 
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;B" title="B" /> = the cognitive phenotypes of an individual, including and measuring behaviors
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;C" title="C" /> = the connectome of an individual
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" />=  the developmental history of an individual
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;E" title="E" /> = the environment of the individual
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;G" title="G" />, the genome of an individual (including epigenetics).

## Connectal coding theories

- can use hypothesis testing to answer some of these questions
- need test statistic and p-value
- search for signal subgraph (small set of nodes and edges to confer most of the info)
