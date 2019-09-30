# Hermaphrodite C. Elegans Neuronal Wiring Data Set ([link](https://www.wormatlas.org/neuronalwiring.html#NeuronalconnectivityII))

- By L.R. Varshney, B.L. Chen, E. Paniagua, D.H. Hall and D.B. Chklovskii

- For more descriptions, see "Structural properties of the C. elegans neuronal network" PLoS Comput. Biol. Feb 3, 2011 3:7:e1001066 (doi:10.1371/journal.pcbi.1001066) by Chen, Paniaqua, Hall and Chklovskii

- This data set is chosen for HSBM because there's an existing [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3098222/) talking about the hierarchical structure of C. elegans using steps like subgraph clustering and motif comparison, which makes this paper a good evaluation standard for HSBM project. More importantly, it offers the biological interpretation of the result, which makes the analysis valid, compared to the original HSBM paper, which doesn't have any discussion of the biological implication of the result and has no too much credibility from this perspective.

## 2.1 Connectivity Data

- C. elegans wiring diagram
- Method: electron micrographs (EM)
- 280 nonpharyngeal neurons (CANL/R were excluded because of the unknown function and that they hav e no obvious synapses)
- 6396 chemical synapses, 890 electrical junctions, 1410 neuromuscular junctions

- Compared to the previous version, add dorsal side of the mid-body
- Ambiguities still exit (sub-lateral, canal-associated lateral, mid-body dorsal cords)

- Labels
    - N1, N2: neuron 1's name, neuron 2's name
    - Type: type of synapse
        - "send" means from N1 to N2
        - distinguish Sp(send to more than one post-synaptic neurons) and S(just send); Rp (receive polu) vs R(just receive)
        - Some Sp may be labeled as S but it follows that S+Sp=R+Rp
    - Nbr: number of synapse between this neuron pair

## 2.2 Neuron description (neuron types)

- Neuron position, synapse position
- 1D information (the position is defined as the neuron's projection onto the anterior-posterior axis)
- LIMIT: the posterior data was from 2 animals (1 hermaphrodite and 1 make)
    - due to the scarcity of high power EM on the dorsal side

- Labels
    - Neuron: name of neuron
    - Soma position, soma region
    - Span: short spans(S) (<25% of the worm body) vs long (L)
    - Ambiguity: what kind of the ambiguity
    - Additonal labels about the position