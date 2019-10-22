# Summary
### [*Network Classification with Applications to Brain Connectomics* Relion et al, 2019, arXiv](https://arxiv.org/abs/1701.08140)
#### Casey Weiner  
#### Last 10 Pages  

##### Page 21:
> Test proved most accurate for COBRE instead of UMich data.

##### Page 22:
> Lambda has little influence on sparsity while p has most influence on sparsity. p < 1 means nodes not selected. Accuracy generally decreased as solutions became more sparse. Choosing parameters by cross-validation more often than not introduces too many noise variables. "One standard error rule": choice is never more than a standard deviation away from the best cross validation accuracy. For SVM, in most cases local kernel methods were as accurate as random guessing. 

##### Page 23:
> Different sample sizes and noise levels likely accounts for the better accuracy on the COBRE data. COBRE data set also more homogeneous thus easier to find pattern. However, still accurate results on both data sets. SVM with hinge perform well too, generally better than methods with logistic loss. Could add new penalty to increase accuracy closer to this but this would make current method deviate from SVM + L1 and solutions based on logistic loss are generally considered more stable. 

##### Page 24:
> In the COBRE data, accuracy can be achieved with a fairly small number of edges due to homogeneity. With the UMich data, there needs to be more due to the noise. In any case, method uses less edges than other methods. In connectomics it is difficult to identify significant variables because of small sample sizes. Thus, *stability selection* is employed, where the data is randomly split multiple times and variables are picked from the groups. This is most relevant to sparse solutions and is not sensitive to initial choice of tuning parameters, and changing these only lightly affects the ordering of variables with the largest likelihoods of being picked. 

##### Page 25:
> Accuracy increased rapidly with respect to the common logarithm of the number of edges.

##### Page 26:
> The results on the UMich data may be less reliable since the upper bound for the number of falsely selected variables is over 50% higher for UMich compared to COBRE. Nonetheless, the default mode network was often selected in both. This network has been consistently implicated with Schizophrenia and other psychiatric disorders. This is a possible sign of psychopathology. In the COBRE data set, edges were selected also from the "fronto-parietal task control region" of the brain, previously linked to Schizophrenia. 

##### Page 27:
> This agrees with the findings of other academics, which demonstrates that the robustness of the methodology is promising. The uncertain system, salience system, and sensory/somatomotor hand regions are all stand outs in the UMich data and are all implicated in connected to schizophrenia. Accuracy is still respectable when one data set is used to train and another to test as opposed to same data set for training and testing. Consistently inactive nodes are mostly clustered in two coherent regions.

##### Page 28:
> This is a methodology for classifying graphs with labeled nodes and associated responses. This method is graph aware, selects a sparse set of nodes *and* edges, and it is general in that it does not rely on the "spatial structure" of the brain. Method is computationally efficient because it is convex optimization with efficient algorithms. The increased speed of this method also means that larger data sets can be analyzed. The time this algorithm takes to run depends on the number of *active* nodes, not the number of *total* nodes. Regions labeled active agree with scientific/biological consensus, the predictions were fairly accurate, and the machine learning technique involved the traditional practice of splitting one set of data into a test set and a training set. Differences in the two data sets may reflect real life differences or type 2 errors. Preprocessing can change the effectiveness of this method. Aimed to mitigate this by using ranks. Another option is to compare the results on images from multiple different preprocessing pipelines and with different values for measures of connectivity. This method's independence of choice and computation efficiency make it a prime candidate for use.

##### Page 29:
> References

##### Page 34
> Use of Algorithm 2 from this paper, as well as proof for the bounds on the Frobenius norm error.
