# Connectal coding ([Jovo 2019](https://www.sciencedirect.com/science/article/abs/pii/S0959438818301430))

> Connectal coding: discovering the structures linking cognitive phenotypes to individual histories (Jovo 2019)

- Introduce in clear terms the definition of connectomes and its value

## 1. Modeling brains as networks

### History

- Ramon y Cajal: neuron doctrine
- McCulloch and Pitts: Brain network --> Turing-complete
- Little-Hopfield networks: can be used to store information
- Connectome: introduced by Sporns el al. and Hagmann

### Definition

- Vertices: Biophysical entitiry with certian constraints
    - Spatial extent
    - Spatial resolution
    - Type
    - Developmental stage

- Edges: the presence/absence of a connection between node pairs with constraints
    - Communication types
    - Temporal duration

### Representation

- 2D array ("adjacency matrix")
    - But not an accurate matrix since rows are related to columns

- Additional features
    - Edge weights (like synapse number)
    - Senmantical label (node's type)
    - Weight of connectomes population (weight of entire graph)

- What is not a connectome
    - Nodes without definitions of vertices (like the shape of neurons)
    - Correlation between a pair of nodes can't be the edge (Disagreement here)

- Scales
    - Could be represented in different scales (cell, regions)
    - Each scale could be an adequate model
        - fMRI: many confounders

## 2. Example estimated connectomes

- Visualization (see reference [29-31])
- Sorting
    - By regions or types
    - By degree
- Difference edge types, different colors

### A. C. elegans
- Complete connectome
- Nodes represent neurons
- Edge types
    - Chemical synapses
    - Gap junction (?)
- Weight
    - Volume of synapse
- 2 sexes
- Method of estimation
    - Manually tracing axons and dendrites
    - Identifying synpase
    - Nanoscale electron micrographs

### B. Drosophila
- Serial electron microscopy
- Only chemical synapse
- Directed graph

### C. Mouse
- dMRI (undirected)
- Weights: number of tracts between regions
- Normalization (don't know the actual weights)
    - Rescale to [0,1]
    - Log scale

### D. Human
- fMRI
    - Pearson correlation matrices
    - Normalization: rank ---> [0,1]
- dMRI: normalization same as mouse

## 3. The puerpose of brain codes
- Neural activity: brain's representation of info
- Connectivity: brain's storage of info

## 4. The role of connectomes in connectal coding
- Link structures to phenotypes
    - Individual histories
    - Cognitive phenotypes

- Value: generating hypothese about connectotypes