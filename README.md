# mlr_GSoC2015
This is the test for GSoC 2015 [project mlr](https://github.com/berndbischl/mlr/wiki/GSOC-2015:-Implement-several-ensemble-SVMs-in-mlr)

#### Introduction
test.R is a simple learner that uses RBF kernerl-based SVM and K-means for binary classification.
* The learner sets three parameters: C, the regularization parameter for the SVM; sigma for the RBF kernel parameter and n for the number of clusters
* The learner first uses kmeans to cluster the training data then fits a RBF kernel SVM to each cluster
* The learner make prediction by simply determining for a given test point the cluster it belongs to and use the associated SVM model to predict.
* The learner can easily be coded as a set of functions.

The dataset used is the breast-cancer_scaled.txt in the format of [libSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).

```R
#### Demo
Here is what I got from `source` test.R
> source("test.R")
# load library
Loading required package: ParamHelpers
Loading required package: BBmisc
Loading required package: ggplot2
# read data: breast-cancer_scaled
# set parameters
## n = 3
## C = 5
## sigma = 0.05
# perform kmeans clustering
# perform RBF kernel SVM
##       for cluster 1
##       for cluster 2
##       for cluster 3
# Do prediction
# Show prediction results:
ksvmPred   2   4
       2 217   6
       4   7 111
```
