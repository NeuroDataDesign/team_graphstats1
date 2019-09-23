Summary for NETWORK CLASSIFICATION WITH APPLICATIONS TO BRAIN CONNECTOMICS,
Arroyo-Reli ´ on et al, 2019, arXiv

Casey Weiner

Page 11:

The algorithm to minimize the coefficients of the B matrix is a combination of
proximal algorithms and alternating direction of multipliers (ADMM). The
proximal algorithms are used to solve the entire problem, however running the
proximal algorithms requires further optimization to calculate a required
proximal operator, which is found with ADMM. The proximal algorithm uses the
descent direction of the differentiable loss function to calculate a new
iteration of B.

Page 12:

The iterations of the proximal algorithm require optimization of a separate
function involving B, thus ADMM is required. ADMM performs gradient descent on a
convex optimization problem by introducing new multipliers U and V.

Page 13:

Specific parameters of the ADMM process, including the constant mu, can be tuned
to provide accurate results. The algorithm runs until a positive tolerance value
is reached. ADMM also allows easy adding of additional penalties to the proximal
algorithm. This algorithm can also be setup differently, such that now
optimization is done over the entire set of Bs.

Page 14:

This formulation allows for a much easier and simpler problem, with a closed
form solution. The two main assumptions on the loss function are centered around
the convexity of the loss function around B\* and bounding the size of the
entries in the loss Hessian.

Page 15:

The first assumption can really be called Restricted Strong Convexity, where the
magnitude of the difference between the Laplacian of L at B and at B\* is less
than or equal to the magnitude of the difference between B and B\* times a
constant. The second assumption can be called Irrepresentability and it states
that there exists a constant that would make the maximum value of the 2D norm of
the sum of the 2D norms of entries raised the N power be negative. There is also
an assumption on the first derivative of the loss function, which equates to a
bound on the maximum value of the 2D norm of the derivative of l at B\*. there
is another assumption that the score function, the maximum probability that the
infinite dimensional norm of the derivative of l is greater than a value t, is
less than two times the Gaussian(t,0,sigma).

Page 16:

The portion of the penalty associated with p allows for sparse solutions. At
least one edge attached to an active node must have non-zero weight. The
methodology utilized is useful in that it works well for datasets where the
sample size is small but the number of nodes is large.

Page 17:

To test the method, an fMRI data set was used in which a stochastic block model,
utilizing random values from a normal distribution, was utilized to create the
weights of the edges, which were meant to match the weights from the datasets.
The networks were undirected.

Page 18:

The AUC of the ROC curve was used to select predictive nodes and edges. The
probability of a false positive and are true positive are calculated with Bayes’
Rule. The prediction accuracy of the methods was defined by finding the most
accurate tuning parameter with cross-validation using the training data and then
computing the test error.

Page 19:

A support vector machine was also used for comparison and the classification
error of the support vector machines was also calculated. Linear kernels were
used for both SVMs. Also considered independent screening method for selection
of variables through t-test. Also compared results to the methodology of signal
subgraphs.

Page 20:

Clearly all methods would improve as more differentiating edges are included in
the dataset, but the method formulated has the best results for all except when
all nodes are active. The method was applied to the COBRE Schizophrenic data
set.
