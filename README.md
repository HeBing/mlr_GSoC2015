# mlr_GSoC2015
This is the test for GSoC 2015 [project mlr](https://github.com/berndbischl/mlr/wiki/GSOC-2015:-Implement-several-ensemble-SVMs-in-mlr):
* `test.R`: a simple learner that uses RBF kernel SVM and K-means for binary classification
* `CSVM.R`: clustered SVM implemented using `mlr` [[ref]](http://jmlr.org/proceedings/papers/v31/gu13b.html)
* `CSVM_realdata.R`: reproduce CSVM results in [[ref]](http://jmlr.org/proceedings/papers/v31/gu13b.html) for Synthetic dataset (Section 4.1), [SVMGUIDE1](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#svmguide1) and [IJCNN1](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1).

#### `test.R`
`test.R` is a simple learner that uses RBF kernerl-based SVM and K-means for binary classification.
* The learner sets three parameters: `C`, the regularization parameter for the SVM; `sigma` for the RBF kernel parameter and `n` for the number of clusters
* The learner first uses kmeans to cluster the training data then fits a RBF kernel SVM to each cluster
* The learner makes prediction by simply determining for a given test point the cluster it belongs to and use the cluster-specific SVM model to make prediction.
* The dataset used is the breast-cancer_scaled.txt in the format of [libSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).

```R
> source("test.R")
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
# Accuracy

```

#### `CSVM.R`
`CSVM.R` implements the clustered SVM learner in [[ref]](http://jmlr.org/proceedings/papers/v31/gu13b.html). It contains three functions:
* `trainCSVM()` takes `C` local regularization parameter for SVM, `lambda` global regularization parameter for CSVM, and `k` number of clusters as well as `feature` for train data feature and `label` for train data response as input and return a list of trained K-means model and SVM models defined by `mlr`
* `predictCSVM()` takes outputed list from `trainCSVM()` and `newdata` the test data feature (excluding response) as input and return a vector of predictions.
* `transformCSVM()` a low-level utility function that reduces the CSVM optimization problem to a standard SVM problem (see formula (5) and (6) in [[ref]](http://jmlr.org/proceedings/papers/v31/gu13b.html))

#### `CSVM_realdata.R`
`CSVM_realdata.R` reproduces the results for the Synthetic Dataset (section 4.1), SVMGUIDE1, and IJCNN1. Note it may take a while (10+ minutes) to complete the IJCNN1 dataset example (49,990 train cases and 91,701 test cases); I am working on speeding it up using C/C++ as underlying core programs.

