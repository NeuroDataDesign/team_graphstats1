# Decision Forests for Classification, Regression, Density Estimation, Manifold Learning, and Semi-Supervised Learning

Authors: A. Criminisi, J. Shotton, and E. Konukoglu

Link: <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/decisionForests_MSR_TR_2011_114.pdf>

## Abstract

- propose single model for learning tasks
- demonstrate margin-maximizing properties
- intro density forests
- propose algorithms for sampling forest generating model
- introduce manifold forests
- propose new forest based algorithms for active learning

## 1 - Overview and scope

### 1.1 - A brief literature survey

- "C4.5" algorithm: trains optimal decision trees from data
  - (from wiki) C4.5 builds trees using information entropy, finds highest normalized information gain and splits on that attribute
- other work uses individual entities to yield greater accuracy
- random decision forest splits on random features rather than chosen ones
- RF samples training data by bagging

### 1.2 - Outline

- overview of paper

## 2 - The random decision forest model

- examples of machine learning tasks
  - classification: recognizing type of category
  - regression: predicting price of house as function of distance
  - learned density: detecting abonormalities in a medical scan under learned density
  - manifold learning: capturing intrinsic variability in size and shapes of brains
  - semisupervised: interactive image segmentation
  - active learning: learning general rule for detecting tumors in images
- __aim of chapter: present model of RF that tackles all these problems__

### 2.1 - Background and notation

- tree is hierarchy of nodes and edges
- decision tree makes decisions
- features are selected from a subset of features of interest with function <img src="https://latex.codecogs.com/svg.latex?\inline&space;\phi(\mathbf{v})" title="\phi(\mathbf{v})" />
- leaf nodes contain a predictor (classifier or regressor) which associates an output (class label) to the input <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{v}" title="\mathbf{v}" />
- <img src="https://latex.codecogs.com/svg.latex?\inline&space;S_1" title="S_1" /> refers to training points reaching node 1, <img src="https://latex.codecogs.com/svg.latex?\inline&space;S_1^L" title="S_1^L" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;S_1^R" title="S_1^R" /> are subsets left and right to children. In binary trees, <img src="https://latex.codecogs.com/svg.latex?\inline&space;S_j&space;=&space;S_j^L&space;\cup&space;S_j^R" title="S_j = S_j^L \cup S_j^R" />, <img src="https://latex.codecogs.com/svg.latex?\inline&space;S_j^L&space;\cap&space;S_j^R&space;=&space;\emptyset" title="S_j^L \cap S_j^R = \emptyset" />, <img src="https://latex.codecogs.com/svg.latex?\inline&space;S_j^L=&space;S_{2j&plus;1}" title="S_j^L= S_{2j+1}" />, and <img src="https://latex.codecogs.com/svg.latex?\inline&space;S_j^R=&space;S_{2j&plus;2}" title="S_j^R= S_{2j+2}" />
- ground truth labels are chosen to minimize and energy function
- randomness is injected during the training process, and deterministic once trees are fixed
- gain of information is computed using the following formula:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I&space;=&space;H(S)&space;-&space;\sum_{i&space;\in&space;\{1,&space;2\}}&space;\frac{|S^j|}{|S|}&space;H(S^i)" title="I = H(S) - \sum_{i \in \{1, 2\}} \frac{|S^j|}{|S|} H(S^i)" />
</p>

- the Shannon entropy is defined as

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?H(S)&space;=&space;-\sum_{c&space;\in&space;\mathcal{C}}&space;p(c)&space;\log(p(c))" title="H(S) = -\sum_{c \in \mathcal{C}} p(c) \log(p(c))" />
</p>

- for continuous distributions, the differential entropy of a <img src="https://latex.codecogs.com/svg.latex?\inline&space;d" title="d" />-variate Gaussian is

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?H(S)&space;=&space;\frac{1}{2}&space;\log&space;\left((2&space;\pi&space;e)^d&space;|\Lambda&space;(S)&space;|&space;\right&space;)" title="H(S) = \frac{1}{2} \log \left((2 \pi e)^d |\Lambda (S) | \right )" />
</p>

- fitting Gaussian to the data, large overlap is associated with low info gain and peakier Gaussians corresponds to high information gain.
- __summary: Cateogrical data - Split, calculate discreate probabilities, calculate information gain; Continuous data - Split, fit gaussian, calculate probabilities, calculate information gain__

### 2.2 - The decision forest model

- each split node <img src="https://latex.codecogs.com/svg.latex?j" title="j" /> is associated with a binary split function

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?h(\mathbf{v},&space;\mathbf{\theta}_j)&space;\in&space;\{&space;0,&space;1&space;\}" title="h(\mathbf{v}, \mathbf{\theta}_j) \in \{ 0, 1 \}" />
</p>

- the model parameters for are <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{\theta}&space;=&space;(\mathbf{\phi},&space;\mathbf{\psi},&space;\mathbf{\tau})" title="\mathbf{\theta} = (\mathbf{\phi}, \mathbf{\psi}, \mathbf{\tau})" />
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{\psi}" title="\mathbf{\psi}" /> actually splits the data
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{\tau}" title="\mathbf{\tau}" /> captures thresholds for inequalities
  - <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{\phi}" title="\mathbf{\phi}" /> selects the features out of <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{v}" title="\mathbf{v}" />
- Linear weak learner
  - given that <img src="https://latex.codecogs.com/svg.latex?[\cdot]" title="[\cdot]" /> is the indicator function, this model is

  <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?h(\mathbf{v},&space;\mathbf{\theta}_j)&space;=&space;[\tau_1&space;>&space;\mathbf{\phi}(\mathbf{v})&space;\cdot&space;\mathbf{\psi}&space;>&space;\tau_2]" title="h(\mathbf{v}, \mathbf{\theta}_j) = [\tau_1 > \mathbf{\phi}(\mathbf{v}) \cdot \mathbf{\psi} > \tau_2]" />
  </p>

  - Stumps: Axis-aligned weak learners
- Nonlinear weak learner
  - Given that <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{\psi}&space;\in&space;\mathbb{R}^{3&space;\times&space;3}" title="\mathbf{\psi} \in \mathbb{R}^{3 \times 3}" /> is a matrix representing the conic section, the conic section is

  <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?h(\mathbf{v},&space;\mathbf{\theta}_j)&space;=&space;\left[&space;\tau_1&space;>&space;\mathbf{\phi}^\top(\mathbf{v})\,\mathbf{\psi}\,\mathbf{\phi}&space;(\mathbf{v})&space;>&space;\tau_2&space;\right]" title="h(\mathbf{v}, \mathbf{\theta}_j) = \left[ \tau_1 > \mathbf{\phi}^\top(\mathbf{v})\,\mathbf{\psi}\,\mathbf{\phi} (\mathbf{v}) > \tau_2 \right]" />
  </p>

  - The number of degrees of freedom of the weak learner influences forest generalization properties
- If the information gain at the <img src="https://latex.codecogs.com/svg.latex?j^\mathrm{th}" title="j^\mathrm{th}" /> split node is <img src="https://latex.codecogs.com/svg.latex?\inline&space;I_j&space;=&space;I(S_j,&space;S_j^L,&space;S_j^R,&space;\mathbf{\theta}_j)" title="I_j = I(S_j, S_j^L, S_j^R, \mathbf{\theta}_j)" />, then the information gain objective function is maximized to compute the optimal spit. That is,

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{\theta}_j^*&space;=&space;\arg&space;\max_{\mathbf{\theta}_j}&space;I_j" title="\mathbf{\theta}_j^* = \arg \max_{\mathbf{\theta}_j} I_j" />
</p>

- randomness is injected during the training phase via
  - bagging
  - randomized node optimization
- if <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathcal{T}" title="\mathcal{T}" /> is the set of all possible <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{\theta}" title="\mathbf{\theta}" /> when training the <img src="https://latex.codecogs.com/svg.latex?j^\mathrm{th}" title="j^\mathrm{th}" /> split node. Each split is optimized via

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{\theta}_j^*&space;=&space;\arg&space;\max_{\mathbf{\theta}_j&space;\in&space;\mathcal{T}_j}&space;I_j" title="\mathbf{\theta}_j^* = \arg \max_{\mathbf{\theta}_j \in \mathcal{T}_j} I_j" />
</p>

- so, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\rho" title="\rho" /> is introduced where <img src="https://latex.codecogs.com/svg.latex?\inline&space;\rho&space;=&space;|\mathcal{T}_j|" title="\rho = |\mathcal{T}_j|" />. Here, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\rho&space;=&space;|\mathcal{T}|" title="\rho = |\mathcal{T}|" /> indicates that all the trees are identical and <img src="https://latex.codecogs.com/svg.latex?\inline&space;\rho&space;=&space;1" title="\rho = 1" /> means there is no randomness in the system.
- the probabilitic leaf predictor model for the <img src="https://latex.codecogs.com/svg.latex?\inline&space;t^\mathrm{th}" title="t^\mathrm{th}" /> tree is <img src="https://latex.codecogs.com/svg.latex?\inline&space;p_t(c|\mathbf{v})" title="p_t(c|\mathbf{v})" />. In regression, the output is continuous and the leaf predictor model is a posterior of the desired continuous variable. In classification trees, it is a point estimate instead.
- tree testing is done in parallel
- in classification, forest prediction is a simple averaging operation:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p(c|\mathbf{v})&space;=&space;\frac{1}{T}&space;\sum_{t=1}^T&space;p_t(c|\mathbf{v})" title="p(c|\mathbf{v}) = \frac{1}{T} \sum_{t=1}^T p_t(c|\mathbf{v})" />
</p>

- or, given a partition function <img src="https://latex.codecogs.com/svg.latex?\inline&space;Z" title="Z" />, multiply tree outputs (even though trees are not statistically independent):

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p(c|\mathbf{v})&space;=&space;\frac{1}{Z}&space;\prod_{t=1}^T&space;p_t(c|\mathbf{v})" title="p(c|\mathbf{v}) = \frac{1}{Z} \prod_{t=1}^T p_t(c|\mathbf{v})" />
</p>

- both averaging and taking the product are heavily influenced by most confident, most informative trees
- hard limit (depth of trees hyper-parameter)
- minimum information gain
- too few points in a class

#### Summary of key model parameters

- parameters that most influence decision forest are
  - the forest size <img src="https://latex.codecogs.com/svg.latex?\inline&space;T" title="T" />
  - the maximium allowed tree depth <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" />
  - the amount of randomness (controlled by <img src="https://latex.codecogs.com/svg.latex?\inline&space;\rho" title="\rho" />) and its type
  - the choice of weak learner model
  - the training objective function
  - the choise of features
- these parameters affect accuracy of its confidence (forest predictive accuracy) and computational efficiency
- very deep trees lead to overfitting

## 3 - Classification forests

### 3.1 - Classification algorithms in the literature

- SVM is most commonly used as it guarantees maximium-margin separation
- boosting builds strong classifiers from linear combination (like Adaboost)
- neither extend to multiclass problems
- classification forests show good generalization, even with high dimensional data

### 3.2 - Specializing the decsion foret model for classification

- the classification problem can be summarized as previously stated

### 3.3 - Effect of model parameters

- suppose shallow trees (<img src="https://latex.codecogs.com/svg.latex?\inline&space;D=2" title="D=2" />)
- increasing forest size from <img src="https://latex.codecogs.com/svg.latex?\inline&space;T=1" title="T=1" /> to <img src="https://latex.codecogs.com/svg.latex?\inline&space;T=200" title="T=200" /> produces smoother posteriors
- decision trees are better than SVM or boosting because it can handle both binary and multi-class problems
- tree depth increases overall prediction confidence
- large values of <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" /> tend to overfitting
- values of <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" /> is a function of the problem complexity
- for a fixed weak learner, increasing <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" /> increases the confidence of the output
- axis alignned tests are efficient to compute so the choice of accuracy and efficency are a tradeoff
- larger randomness reduces blocky artifacts of axis-aligned weak learner but redduces overall confidence.
- however, larger weak learners have a higher associated parameter space

### 3.4 - Maximum-margin properties

- formally, consider weak learners to be vertical lines only for a two class problem, _i.e._

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\inline&space;h(\mathbf{v},&space;\mathbf{\theta}_j)&space;=&space;\left[&space;\phi(\mathbf{v})&space;>&space;\tau&space;\right&space;]&space;\quad&space;\mathrm{with}&space;\quad&space;\phi(\mathbf{v})&space;=&space;x_1" title="h(\mathbf{v}, \mathbf{\theta}_j) = \left[ \phi(\mathbf{v}) > \tau \right ] \quad \mathrm{with} \quad \phi(\mathbf{v}) = x_1" />
</p>

- the optimal separting line at position <img src="https://latex.codecogs.com/svg.latex?\inline&space;\tau^*" title="\tau^*" /> is

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\tau^*&space;=&space;\arg&space;\min_\tau&space;|p(c&space;=&space;c_1\,|\,x_1&space;=&space;\tau)&space;-&space;p(c&space;=&space;c_2\,|\,x_1&space;=&space;\tau)|" title="\tau^* = \arg \min_\tau |p(c = c_1\,|\,x_1 = \tau) - p(c = c_2\,|\,x_1 = \tau)|" />
</p>

- available test parameters are sampled uniformly, then forest posteriors behave linearly
- if posteriors are are constant, then the optimal separation is in the middle of the gap (margin-maximization)
- as <img src="https://latex.codecogs.com/svg.latex?\inline&space;T&space;\rightarrow&space;\infty" title="T \rightarrow \infty" />, the desired max-margin behavior is produced as in SVM
- the combination is fully probabibilistic
- more randomness is not guaranteed to split the data perfectly and leads to lower information gain
- randomness parameter similar to the slack variable in SVM
- little randomness similar to SVMs
- using linear weak-learners produce globally nonlinear classification
- the choice of weak learner affects the optimal hard separating surface
- conic learners shape the uncertainty region in a curved fasion
- in bagging, 50% of the training data is sampled with replacement
- randomly set the parameter (RNO) and bagging together were used leading to smoother posteriors with optimal boundaries not maximum margin
  - bagging improves training speed
  - SNO is used because it uses all training data and allows to control maximum-margin by changing randomness parameter <img src="https://latex.codecogs.com/svg.latex?\inline&space;\rho" title="\rho" />

### 3.5 - Comparisons with alternative algorithms

- against boosting
  - forests vs. ModestBoost with shallow stumps, axis-aligned weak learners
  - forests produced smooth, probabilistic output while boosting produces a hard output
- against SVM
  - all four class examples are nicely separable
  - forests and SVM separate well, while forests provide uncertainty information
  - SVM produces equal confidence per pixel

### 3.6 - Human body tracking in Microsoft Kinect for XBox 360

- 31 different body part classes
- pixel <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{p}&space;\in&space;\mathbb{R}^2" title="\mathbf{p} \in \mathbb{R}^2" /> with associated feature vector <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{v}(\mathbf{p})&space;\in&space;\mathbb{R}^d" title="\mathbf{v}(\mathbf{p}) \in \mathbb{R}^d" />
- feature vector is collection of depth differences
- use axis-aligned weak learner

## 4 - Regression forests

- labels are continuous

### 4.1 - Nonlinear regression in the literature

- least squares fits a linear regressor to minimize some error
  - limitation is its linear
  - also is sensitive input noise
- RANSAC is a poopular technique to achieve regression via randomization (outputs are non probabilistic) -> regression forests are an extention of this
- SVR is simple model
- nonprobabilistic regression forests also exist

### 4.2 - Specializing the decision forest model for regression

- learn a general mapping with previously unseen independent data with the correct continuous prediction
- given a multivariate input <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{v}" title="\mathbf{v}" />, wish to associate multivariate label <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{y}&space;\in&space;\mathcal{Y}&space;\subset&space;\mathbb{R}^n" title="\mathbf{y} \in \mathcal{Y} \subset \mathbb{R}^n" /> -> estimate <img src="https://latex.codecogs.com/svg.latex?\inline&space;p(\mathbf{y}&space;|&space;\mathbf{v})" title="p(\mathbf{y} | \mathbf{v})" />
- a polynomial model can be used the estimate the class posterior (pre-stored in classification) -> simple and captures many relationships
- forest output is average of all tree outputs
- forest training happens by optimiizing over a training set
- main difference is the form of the objective function <img src="https://latex.codecogs.com/svg.latex?\inline&space;I_j" title="I_j" />

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_j&space;=&space;\sum_{\mathbf{v}&space;\in&space;S_j}&space;\log&space;(|\Lambda_\mathbf{y}&space;(\mathbf{v})|)&space;-&space;\sum_{i&space;\in&space;\{\mathrm{L},&space;\mathrm{R}\}}&space;\left(&space;\sum_{\mathbf{v}&space;\in&space;S^i_j}&space;\log&space;(|\Lambda_\mathbf{y}&space;(\mathbf{v})|)&space;\right&space;)" title="I_j = \sum_{\mathbf{v} \in S_j} \log (|\Lambda_\mathbf{y} (\mathbf{v})|) - \sum_{i \in \{\mathrm{L}, \mathrm{R}\}} \left( \sum_{\mathbf{v} \in S^i_j} \log (|\Lambda_\mathbf{y} (\mathbf{v})|) \right )" />
</p>

- the function inside the <img src="https://latex.codecogs.com/svg.latex?\inline&space;\log" title="\log" /> is the conditional covariance matrix from probabilistic linear fitting
- the error or fit objective function is LS between single-variate output and mean output for all training points
- split by binary weak learner (three types considered -> axis-aligned, oriented hyperplane, quadratic)

### 4.3 - Effect of model parameters

- increasing forest size produces moother class posteriors and smoother mean curves
- deeper trees may overfit the data
- uncertainty increases away from training data

### 4.4 - Comparison with alternative algorithms

- against Guassian processes
  - behavior is similar
  - shape of uncertainty determined by the prediction model
  - GP leads to overconfident predictions on ambiguous (noisy) data

### 4.5 - Semantic parsing of 3D computed tomography scans

- how to place a bounding box within the image
- each voxel takes in image votes for where it thinks the organ should be (relative displacement vectors)
- for a voxel, the feature vector is a collection of differences (density of tissue)
- advantage of forests is interpretability, cluster of points represents pretty good confidence
- regression forests used in localization for full body MRI images

## 5 - Density forests

- find intrinisic nature and structure of unabled data
- problem closely related to clustering

### 5.1 - Literature of density estimation

- $k$-means in the standard, and GMMs are used to approximate complex distributions as a set of simple multivariate Gaussians
- Parzen-Rosenblatt window estimates is kernel based, and $k$-nearest neighbor algorithm is related

### 5.2 - Specializing the forest model for density estimation

- given a set of unlabeled observations we wish to estimate the probability density function from which such data has been generated
- data point is represented as a multidimensional feature response vector
- density forest is a generalization of GMM with multiple hard clustered data paritions and the forest posterior is a combination of tree posteriors (rather than linear combinations of Gaussians)
- define unsupervised entryopy of <img src="https://latex.codecogs.com/svg.latex?\inline&space;d" title="d" />-variate Guassians iis find using

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?H(S)&space;=&space;\frac{1}{2}&space;\log&space;\left(&space;(2&space;\pi&space;e)^d&space;|\Lambda&space;(S)|&space;\right&space;)" title="H(S) = \frac{1}{2} \log \left( (2 \pi e)^d |\Lambda (S)| \right )" />
</p>

- if the cardinality of determinant of the matrix is denoted by <img src="https://latex.codecogs.com/svg.latex?\inline&space;|\cdot|" title="|\cdot|" />, the information gain reduces to

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_j&space;=&space;\log&space;(|\Lambda&space;(S_j)|)&space;-&space;\sum_{i&space;\in&space;\{&space;\mathrm{L},&space;\mathrm{R}\}}&space;\frac{|S^i_j|}{S_j}&space;\log&space;(|\Lambda&space;(S^i_j)|)" title="I_j = \log (|\Lambda (S_j)|) - \sum_{i \in \{ \mathrm{L}, \mathrm{R}\}} \frac{|S^i_j|}{S_j} \log (|\Lambda (S^i_j)|)" />
</p>

- the deteriminant of the covariance matrix is a function of the volume of the ellipsoid corresponding to the cluster
- Guassians are used since they are simple and produce easier distributions to use
- the output of the training points at the leaf node is a multivariate Guassian distribution:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p_t&space;(\mathbf{v})&space;=&space;\frac{\pi_{l&space;(\mathbf{v})}}{Z_t}&space;\mathcal{N}&space;(\mathbf{v};&space;\mathbf{\mu}_{l&space;(\mathbf{v})};&space;\mathbf{\Lambda}_{l&space;(\mathbf{v})})" title="p_t (\mathbf{v}) = \frac{\pi_{l (\mathbf{v})}}{Z_t} \mathcal{N} (\mathbf{v}; \mathbf{\mu}_{l (\mathbf{v})}; \mathbf{\Lambda}_{l (\mathbf{v})})" />
</p>

- to ensure probabilistic normalization, the partition function <img src="https://latex.codecogs.com/svg.latex?\inline&space;Z_t" title="Z_t" /> is defined as

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Z_t&space;=&space;\int_\mathbf{v}&space;\pi_{l&space;(\mathbf{v})}&space;\mathcal{N}&space;(\mathbf{v};&space;\mathbf{\mu}_{l&space;(\mathbf{v})};&space;\mathbf{\Lambda}_{l&space;(\mathbf{v})})\,&space;d\mathbf{v}" title="Z_t = \int_\mathbf{v} \pi_{l (\mathbf{v})} \mathcal{N} (\mathbf{v}; \mathbf{\mu}_{l (\mathbf{v})}; \mathbf{\Lambda}_{l (\mathbf{v})})\, d\mathbf{v}" />
</p>

- forest density is average of all tree densities
- while GMM parameters are learned via Expectation Maximization (EM), parameters of density forest learned via information gain maximization

### 5.3 - Effect of model parameters

- deeper trees lead to further splits and smaller Gaussians
  - tend to "fit to noise" of training data rather than captruing underlying nature of the data
- even if individual trees over-fit, more trees lead to smoother densities
- __since increasing <img src="https://latex.codecogs.com/svg.latex?\inline&space;T" title="T" /> always produces better results (with increased computational costs), we can just set <img src="https://latex.codecogs.com/svg.latex?\inline&space;T" title="T" /> to a "sufficiently large" value with optimizing the value

### 5.4 - Comparison with alternative algorithms

- forests produce must smoother results for forest output
- Parzen and nearest neighbor estimators produce artifacts due to hard choices of windows of number of neighbors
- against GMM EM
  - using more components does not make results any better
  - use of randomness yields improved results
  - EM getting stuck in local minima produce artifacts mitigated in the forest model
  - under random restart GMM, the cost of the model is <img src="https://latex.codecogs.com/svg.latex?\inline&space;R&space;\times&space;T&space;\times&space;G" title="R \times T \times G" /> where <img src="https://latex.codecogs.com/svg.latex?\inline&space;R" title="R" /> is the number of random restarts, <img src="https://latex.codecogs.com/svg.latex?\inline&space;T" title="T" /> is the number of Gaussian components, and <img src="https://latex.codecogs.com/svg.latex?\inline&space;G" title="G" /> is the cost of evaluating the feature vector under each individual Guassian
  - under the density forest with <img src="https://latex.codecogs.com/svg.latex?\inline&space;T" title="T" /> trees of maximum depth <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" /> has cost <img src="https://latex.codecogs.com/svg.latex?\inline&space;T&space;\times&space;G&space;&plus;&space;T&space;\times&space;D&space;\times&space;B" title="T \times G + T \times D \times B" /> where <img src="https://latex.codecogs.com/svg.latex?\inline&space;B" title="B" /> is the cost of a binary test at the split node
  - since <img src="https://latex.codecogs.com/svg.latex?\inline&space;B" title="B" /> is often really small, can disregard last term; this corresponds to cost of a single GMM

### 5.5 - Sampling from the generative model

- describe algorithm for sampling random data under the learned model
- the cost of sampling from density forests is equivalent to sampling from random-restart GMM

### 5.6 - Dealing with non-function relations

- density forests are better equipped than regression forests for ambigious training data (not one-to-one between inputs)
- before, regression forests were calculating posteriors around a central region, but now data points are treated as a pair with both dimensions treated as input features
- the the joint generative density function is estimated <img src="https://latex.codecogs.com/svg.latex?\inline&space;p(x,&space;y)" title="p(x, y)" /> and calculated as

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p(x,&space;y)&space;=&space;\frac{1}{T}&space;\sum_{t=1}^T&space;p_t&space;(x,&space;y)" title="p(x, y) = \frac{1}{T} \sum_{t=1}^T p_t (x, y)" />
</p>

- with the same individual tree density as before except with the joint density and with mean <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{\mu}_l&space;=&space;(\mu_x,&space;\mu_y)" title="\mathbf{\mu}_l = (\mu_x, \mu_y)" /> and covariance

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Lambda_l&space;=&space;\begin{pmatrix}&space;\sigma^2_{xx}&space;&&space;\sigma^2_{xy}&space;\\&space;\sigma^2_{yx}&space;&&space;\sigma^2_{yy}&space;\end{pmatrix}" title="\Lambda_l = \begin{pmatrix} \sigma^2_{xx} & \sigma^2_{xy} \\ \sigma^2_{yx} & \sigma^2_{yy} \end{pmatrix}" />
</p>

- given an axis-aligned weak learner for the density <img src="https://latex.codecogs.com/svg.latex?\inline&space;p(x,y)" title="p(x,y)" /> node inputs depend on the inputs and not on anything else, and leaf conditional mean and variance <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mu_{y|x,l}&space;=&space;\mu_y&space;&plus;&space;\frac{\sigma^2_{xy}}{\sigma^2_{yy}}&space;(x^*&space;-&space;\mu_x)" title="\mu_{y|x,l} = \mu_y + \frac{\sigma^2_{xy}}{\sigma^2_{yy}} (x^* - \mu_x)" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;\sigma^2_{y|x,l}&space;=&space;\sigma^2_{yy}&space;-&space;\frac{\sigma^4_{xy}}{\sigma^2_{xx}}" title="\sigma^2_{y|x,l} = \sigma^2_{yy} - \frac{\sigma^4_{xy}}{\sigma^2_{xx}}" />, the tree conditional density is

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p(y&space;|&space;x=x^*)&space;=&space;\frac{1}{Z_{t,x^*}}&space;\sum_{l&space;\in&space;\mathcal{L}_{t,x^*}}&space;[y^B_l&space;\leq&space;y&space;<&space;y^T_l]\&space;\pi_l\&space;\mathcal{N}(y;&space;\mu_{y|x,l},&space;\sigma^2_{y|x,l})" title="p(y | x=x^*) = \frac{1}{Z_{t,x^*}} \sum_{l \in \mathcal{L}_{t,x^*}} [y^B_l \leq y < y^T_l]\ \pi_l\ \mathcal{N}(y; \mu_{y|x,l}, \sigma^2_{y|x,l})" />
</p>

- the <img src="https://latex.codecogs.com/svg.latex?\inline&space;Z_{t,x^*}" title="Z_{t,x^*}" /> paramater is found by calculating the conditional CDF difference between the upper and lower bounds of <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" />
- the forest conditional is just the averatge of the individual tree posteriors as before

### 5.7 - Quantitative analysis

- overall error increases after optimal value of depth (overfitting)
- large errors occured because axis-aligned weak learners were used in this case

## 6 - Manifold forests

- related to dimensionality reduction and embedding
- allows for automatic selection of discriminative features
- is part of a more general forest model
- automatic estimation of optimal dimensionality

### 6.1 - Literature of manifold learning

- simpliest algorithm is PCA (linear model)
- isometric feature mapping preserves geodesic distances
- built upon Laplacian eigenmaps (preserve local pairwise distances only)

### 6.2 - Specializing the forest model for manifold learning

- given a set of unlabelled observation, we wish to find a smooth mapping that preserves the observations' relative geodesic distances
- manifold forests built upon density forests (collections of clustering trees)
- manifold forest model requires affinity model for optimal mapping
- randomized node optimization is used again
- information gain is also the same as before
- the predictor model is the same as before
- for a clustering tree, the affinity matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;W^t_ij&space;=&space;e^{-D^t(\mathbf{v}_i,&space;\mathbf{v}_j)}" title="W^t_ij = e^{-D^t(\mathbf{v}_i, \mathbf{v}_j)}" /> with different distance definitions
- the affinity matrix is the average of the tree affinity matrices
- low dimensionall embeddings can be found using a normalized graph-Laplacian matrix
- the nonlinear embeddings can be found by formin matrix with the eigenvector is each column (starting with the first) and mapping each poiont to it's corresponding point after SVD
- can interpolate previously unseen pints in the low dimensional space
- for images, pairwise distances can be calculated by defining in the tree structure itself (in training)
- can define family of features and the tree training process can optimally select features based on the paramaters
- the bottleneck is the solution to the eigven-system

### 6.3 - Experiments and the effect of model parameters

- in binary affinity model, the values of the affinity matrix is binary
- large amount of trees is more robust
- binary model converges more slowly to truth than the Guassian model, but is much faster
- more trees allows for better determination of intrinisic dimensionality of mapped space when plotting magnitudes of eigenvalues

## 7 - Semi-supervised forests

- medium between previous chapters to focus on between supervised (regression and classification) and unsupervised (density and manifold)
- we have a small set of labelled point and a large set of unlabelled ones for training

### 7.1 - Literature on semi-supervised learning

- transductive SVM which maximizes separation in both labelled and unlabelled data

### 7.2 - Specializing the forest model for manifold learning

- given both labelled and unlabelled data, we wish to associate a class label to all unlabelled data
- optimize paramaters by maximizing information gain again
- information gain calculated by adding unsupervised (dependent on all the data) and supervised (dependent on only the labelled data) information gains
- to calculate these see the classification information gain and the density information gains discussed above
- to get posterior, average posteriors of all the trees

### 7.3 - Label propagation in transduction forest

- each leaf stores a different Gasussian with learned MLE for points within
- use Mahalanobis symmetric geodesic distances

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?d(\mathbf{s}_i,&space;\mathbf{s}_j)&space;=&space;\frac{1}{2}&space;\left(&space;\mathbf{d}_{ij}^T&space;\Lambda_{l(\mathbf{v}_i)}^{-1}&space;\mathbf{d}_{ij}&space;&plus;&space;\mathbf{d}_{ij}^T&space;\Lambda_{l(\mathbf{v}_j)}^{-1}&space;\mathbf{d}_{ij}&space;\right&space;)" title="d(\mathbf{s}_i, \mathbf{s}_j) = \frac{1}{2} \left( \mathbf{d}_{ij}^T \Lambda_{l(\mathbf{v}_i)}^{-1} \mathbf{d}_{ij} + \mathbf{d}_{ij}^T \Lambda_{l(\mathbf{v}_j)}^{-1} \mathbf{d}_{ij} \right )" />
</p>

- using this distance matric discourages from cutting across regions of low data density

### 7.4 - Induction from transduction

- infer general probabilistic classification rule for geodesic-based algorithm (geodesic is just spherical distance btw)
- tree posteriors are learned from all class labels
- more trees means smoother posteriors
- semis-supervised forests are not iterative unlike self-training techniques

### 7.5 - Examples, comparisons, and effect of model parameters

- adding more labelled data shows higher predictor confidence
- unlike SVM, the forest capture uncertainty
- can handle multiple classes

## 8 - Random ferns and other forest variants

### 8.1 - Extremly randomized trees

- optimization of each node paramater has been reduced or removed
- increasing randomness produces lower overall prediction confidence and slower convergence of test error

### 8.2 - Random ferns

- same test paramaters used in all nodes at the same tree level
- ferns require deeper forests to split on linearly separable input datasets
- both this and randomized trees are low parametric versions of decisions forests and so overfit much less

### 8.3 - Online forest training

- useful to not have to train learned forests quickly without having to start training from scratch with new labelled training data
- keep new training data fixed and then quickly update corresponding distributions

### 8.4 - Structured-output forests

- can improve on classification when there is inherent dependence among input data
- entangled forests
  - feature feature vector is a function of the data and the output of previous split nodes in the same tree
  - can use both intensities of an image and combinations of class posteriors to show improved generalization with shallow trees
  - learns class spefic context
- decision tree fields
  - combine random decision forests and random fields together in a model
  - different weights are learned using randomized decision trees
  - use approximate MLEs

### 8.5 - Further forest variants

- STAR model is a forest randomly trained on non-binary trees of depth 1
- node harvest is made of multiple single nodes
