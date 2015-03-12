#----------------------------
# 2015-03-12 BH
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

#------------------------#
# read and process data  #
# synthetic dataset:     #
#   four Gaussian,       #
#   Section 4.1 in paper #
#------------------------#
cat("# simulate data: 4 Gaussian\n")
set.seed(20150312)
dat = rbind( mvrnorm(100, mu = c(-2,0), Sigma = diag(rep(0.1,2))),
              mvrnorm(100, mu = c(2,0), Sigma = diag(rep(0.1,2))),
              mvrnorm(100, mu = c(0,-2), Sigma = diag(rep(0.1,2))),
              mvrnorm(100, mu = c(0,2), Sigma = diag(rep(0.1,2))) )
# plot(dat)
dat = data.frame(dat)
dat = cbind(label = gl(2,200),dat)

training.set = sample(1:nrow(dat), round(nrow(dat)/2))
test.set = setdiff(1:nrow(dat), training.set)

#-----------------------#
# set parameters        #
#-----------------------#
cat("# set parameters \n")
k = 2
C = 5
lambda = 5
cat(paste0("## k = ", k,"\n"))
cat(paste0("## C = ", C,"\n"))
cat(paste0("## lambda = ", lambda,"\n"))

#-----------------------#
# clustering            #
#-----------------------#
cat("# perform kmeans clustering\n")
task = makeClusterTask(data = dat[, -1])
learner = makeLearner("cluster.kmeans", par.vals = list(centers = k))
model = train(learner, task, subset = training.set)

#-----------------------#
# reduce clustered SVM  #
#   to a standard SVM   #
#-----------------------#
cat("# reduce clustered SVM to a standard SVM\n")
trainDat = dat[training.set,]
datTilde = data.frame( matrix(0, nrow=nrow(trainDat), ncol=(ncol(trainDat)-1)*(k+1)) )
clusterMembership = model$learner.model$cluster

for(i in 1:nrow(datTilde)) {
  datTilde[i,1:(ncol(trainDat)-1)] = 1/sqrt(lambda)*trainDat[i,-1]
  datTilde[i, (clusterMembership[i]*(ncol(trainDat)-1)+1):((clusterMembership[i]+1)*(ncol(trainDat)-1))] = trainDat[i,-1]
}

datTilde = cbind(label = trainDat$label,datTilde)

clusterKSVM = list()
length(clusterKSVM) = k

for(i in 1:k) {
  cat(paste0("## \t for cluster ",i,"\n"))
  clusterSet =  training.set[clusterMembership == i]
  if( length(unique(dat[clusterSet,1])) == 1) { # one label in this cluster
    clusterKSVM[[i]] = datTilde[clusterSet[1],1]
  } else {
    task2 = makeClassifTask(data = datTilde, target = "label")
    learner2 = makeLearner("classif.ksvm", par.vals = list(kernel = "vanilladot", C = C))
    model2 = train(learner2, task2, subset = clusterSet)
    clusterKSVM[[i]] = model2
  }
}
 
#-----------------------#
# predict               #
#-----------------------#
cat("# Do prediction\n")
clusterPred = predict(model, task = task, subset = test.set)
clusterPred = clusterPred$data$response

# reduce clustered SVM to a standard SVM
testDat = dat[test.set,]
testDatTilde = data.frame( matrix(0, nrow=nrow(testDat), ncol=(ncol(testDat)-1)*(k+1)) )
clusterPred = clusterPred$data$response

for(i in 1:nrow(testDatTilde)) {
  testDatTilde[i,1:(ncol(testDat)-1)] = 1/sqrt(lambda)*testDat[i,-1]
  testDatTilde[i, (clusterPred[i]*(ncol(testDat)-1)+1):((clusterPred[i]+1)*(ncol(testDat)-1))] = testDat[i,-1]
}

testDatTilde = cbind(label = testDat$label,testDatTilde)

ksvmPred = character(length(test.set))

for(i in 1:k) {
  testClusterSet = (1:length(test.set))[clusterPred == i]
  if(length(clusterKSVM[[i]]) == 1) { # if cluster has one label
    ksvmPred[clusterPred == i] = as.character(clusterKSVM[[i]])
  } else {
    tmpPred = predict(clusterKSVM[[i]], newdata = testDatTilde[testClusterSet,])
    ksvmPred[clusterPred == i] =  as.character(tmpPred$data$response)
  }
}

#------------------------#
# prediction results     #
#------------------------#
cat("# Show prediction results:")
ksvmPred = as.factor(ksvmPred)
print(table(ksvmPred, dat[test.set,1]))

plot(dat[test.set,-1], col = as.numeric(ksvmPred))
