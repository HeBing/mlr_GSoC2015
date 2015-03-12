#----------------------------
# 2015-03-12 BH
# train and predict functions for
# clustered support vector machines
# for GSoC 2015
# mlr project: ensemble SVM
# ref: http://jmlr.org/proceedings/papers/v31/gu13b.html
#----------------------------

#---------------------#
# load library        #
#---------------------#
cat("# load library\n")
library(mlr)
library(clue)
library(kernlab)
library(MASS)

#--------------------------------#
# utility function for reducing  #
#   CSVM to a standard SVM       #
#--------------------------------#
transformCSVM <- function(feature, clusterMembership, lambda) {
  stopifnot(is.data.frame(feature) && nrow(feature) == length(clusterMembership) && lambda > 0)

  k = length(unique(clusterMembership))
  datTilde = data.frame( matrix(0, nrow=nrow(feature), ncol=ncol(feature)*(k+1)) )
  for(i in 1:nrow(datTilde)) {
    datTilde[i,1:(ncol(feature))] = 1/sqrt(lambda)*feature[i,]
    datTilde[i, (clusterMembership[i]*(ncol(feature))+1):((clusterMembership[i]+1)*(ncol(feature)))] = feature[i,]
  }
  return(datTilde)
}

#-----------------------#
# clustered SVM train   #
#-----------------------#
# k: num of clusters    #
# C: regularization     #
#   param for SVM       #
# lambda: global        #
#   regularization param#
# feature: data frame   #
# label: vector      #
#-----------------------#

trainCSVM <- function(k, C, lambda, label, feature) {
  stopifnot(is.numeric(k) && is.numeric(C) && is.numeric(lambda))
  stopifnot(k > 0 && C > 0 && lambda > 0)
  stopifnot(class(label) == 'factor')
  stopifnot(is.data.frame(feature) && nrow(feature) == length(label))

  cat("# parameters set to\n")
  cat(paste0("## k = ", k,"\n"))
  cat(paste0("## C = ", C,"\n"))
  cat(paste0("## lambda = ", lambda,"\n"))

  # clustering
  task = makeClusterTask(data = feature)
  learner = makeLearner("cluster.kmeans", par.vals = list(centers = k))
  model = train(learner, task)
  clusterMembership = model$learner.model$cluster

  # CSVM
  featureCSVM = transformCSVM(feature, clusterMembership, lambda)
  datCSVM = cbind(label = label, featureCSVM)
  clusterKSVM = list()
  length(clusterKSVM) = k

  for(i in 1:k) {
    cat(paste0("## \t for cluster ",i,"\n"))
    clusterSet =  (1:nrow(datCSVM))[clusterMembership == i]
    if( length(unique(datCSVM[clusterSet,1])) == 1) { # one label in this cluster
      clusterKSVM[[i]] = datCSVM[clusterSet[1],1]
    } else {
      task2 = makeClassifTask(data = datCSVM, target = "label")
      learner2 = makeLearner("classif.ksvm", par.vals = list(kernel = "vanilladot", C = C))
      model2 = train(learner2, task2, subset = clusterSet)
      clusterKSVM[[i]] = model2
    }
  }

  return(list(clusterModel = model, clusterKSVM = clusterKSVM, lambda = lambda))
}


#-----------------------#
# predict               #
# newdata: data.frame   #
# trainedKSVM: model    #
#   output by trainCSVM #
#-----------------------#
predictCSVM <- function(newdata, trainedKSVM) {
  # browser()
  clusterPred = predict(trainedKSVM$clusterModel, newdata = newdata)
  clusterPred = clusterPred$data$response
  
  newdataCSVM = transformCSVM(newdata, clusterPred, trainedKSVM$lambda)
  newdataCSVM = cbind(label = rep(NA, nrow(newdataCSVM)), newdataCSVM)

  predCSVM = character(nrow(newdataCSVM))
  clusterKSVM = trainedKSVM$clusterKSVM

  for(i in 1:k) {
    testClusterSet = (1:nrow(newdataCSVM))[clusterPred == i]
    if(length(clusterKSVM[[i]]) == 1) { # if cluster has one label
      predCSVM[clusterPred == i] = as.character(clusterKSVM[[i]])
    } else {
      tmpPred = predict(clusterKSVM[[i]], newdata = newdataCSVM[testClusterSet,])
      predCSVM[clusterPred == i] =  as.character(tmpPred$data$response)
    }
  }

  return(predCSVM)
}


