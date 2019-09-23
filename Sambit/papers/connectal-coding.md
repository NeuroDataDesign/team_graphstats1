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
