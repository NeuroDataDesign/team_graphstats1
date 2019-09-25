# HSBM [(Lyzinski 2016) ](https://arxiv.org/abs/1503.02115)

## 0.  **Abstract**

A robust, scalable, integrated method for subcommunity detection and comparison

## 1.  **Introduction**

### a. Network study

-   Basic idea: large graphs composed of loose subgraphs

-   Goal: classifying subcommunity

-   Connectomics (see reference \[4\])

    -   (Cortical column conjecture)

    -   Limited \# of basic computing methods repeat ---&gt;brain's algos

    -   Data to test this conjecture not yet available \[6\]

-   Social network: Friendster

    -   Communication structures affected by larger community

    -   ---&gt; should have repeated structures across subcommunity

### b. Previous studies

-   Community detection

    -   Maximizing modularity and likelihood \[7,8,9\]

    -   Random walks \[10, 11\]

    -   **Spectral clustering** \[12-17\]

        -   Author's previous study \[18\]

        -   Average degree need to be at least order

        -   Perform worse in parse subgraphs (table 1)

    -   **This current study's contribution**

        -   Formal delineation of hierarchical structures in a network

        -   Uncover subgraphs at multi scales

-   Network comparison \[23-29\]

    -   Use method related to \[28\] here (sec 2)

### c. This current study

-   Community detection

    -   Lower-dimensional embedding \[14\]

    -   Adapted spectral clustering \[18\]

        -   Algo 2

-   Community comparison

    -   Nonparametric graph inference \[28\]

        -   Density estimation

-   Multi-sample hypothesis testing \[29\]

-   Better than k-means in hierarchical classification tasks

-   Apply HSBM to (1) drosophila and (2) Friendster

## 2.  **Background**

a. Algorithm 1 (Main algo to detect hierarchical structure for graphs)

-   Graspy [Shell code](https://github.com/neurodata/graspy/blob/18c34bc224b15b93d1d6b809515ac3f8e5733aa5/graspy/models/sbm.py#L497)

-   Adjacency spectral embedding (ASE)

-   Cluster subgraphs using Algo 2

-   Compute ASE for each subgraph to get a matrix with all subgraph's ASE

-   Compute dissimilarities between every two subgraphs

-   Cluster subgraphs based on their motif (e.g. their structure types)

    -   Nonparametric test procedure \[28\]

-   Recurse to the sub-sub level (on the current biggest subgraph)

b. Algorithm 2

## 3.  **2-level HSBM**

-   Analysis of a 2-level synthetic HSBM graph (Fig.2)

    -   8 different blocks (subgraphs)

    -   3 distinct motifs

-   Successfully detect 3 motifs (Fig 3)

-   The actual B matrix (block probability matrix) is slightly different from the value they set, but acceptable

-   Thm 9?

## 4.  **Multilevel HSBM**


## 5.  **Experiments**

-   Drosophila

-   Friendster
