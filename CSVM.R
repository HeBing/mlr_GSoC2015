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
  # browser()
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
  taskCSVM = makeClassifTask(data = datCSVM, target = "label")
  learnerCSVM = makeLearner("classif.ksvm", 
    par.vals = list(kernel = "vanilladot", C = C))
  modelCSVM = train(learnerCSVM, taskCSVM)

  return(list(modelCluster = model, modelCSVM = modelCSVM, lambda = lambda))
}


#-----------------------#
# predict               #
# newdata: data.frame   #
# trainedCSVM: model    #
#   output by trainCSVM #
#-----------------------#
predictCSVM <- function(newdata, trainedCSVM) {

  clusterPred = predict(trainedCSVM$modelCluster, newdata = newdata)
  clusterPred = clusterPred$data$response
  
  newdataCSVM = transformCSVM(newdata, clusterPred, trainedCSVM$lambda)
  newdataCSVM = cbind(label = rep(NA, nrow(newdataCSVM)), newdataCSVM)

  predCSVM = predict(trainedCSVM$modelCSVM, newdata = newdataCSVM)

  return(predCSVM$data$response)
}


