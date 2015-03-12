#----------------------------
# 2015-03-09 BH
# test for GSoC 2015
# mlr project: ensemble SVM
#----------------------------

#---------------------#
# load library        #
#---------------------#
cat("# load library\n")
library(mlr)
library(clue)
library(kernlab)

#-----------------------#
# read and process data #
#-----------------------#
cat("# read data: breast-cancer_scaled\n")
dat = read.table("./data/breast-cancer_scaled.txt", stringsAsFactor = F)
dat[,1] = as.factor(as.character(dat[,1]))
for(i in 2:ncol(dat)) {
  dat[,i] = as.numeric(unlist(strsplit(dat[,i], split = ":"))[(1:(nrow(dat)*2)) %% 2 == 0])
}
colnames(dat)[1] = "response"

training.set = sample(1:nrow(dat), round(nrow(dat)/2))
test.set = setdiff(1:nrow(dat), training.set)

#-----------------------#
# set parameters        #
#-----------------------#
cat("# set parameters \n")
n = 3
C = 5
sigma = 0.05
cat(paste0("## n = ", n,"\n"))
cat(paste0("## C = ", C,"\n"))
cat(paste0("## sigma = ", sigma,"\n"))


#-----------------------#
# clustering            #
#-----------------------#
cat("# perform kmeans clustering\n")
task = makeClusterTask(data = dat[, -1])
learner = makeLearner("cluster.kmeans", par.vals = list(centers = n))
model = train(learner, task, subset = training.set)

#-----------------------#
# train RBF kernel SVM  #
#-----------------------#
cat("# perform RBF kernel SVM\n")
clusterMembership = model$learner.model$cluster
clusterKSVM = list()
length(clusterKSVM) = n

for(i in 1:n) {
  cat(paste0("## \t for cluster ",i,"\n"))
  clusterSet =  training.set[clusterMembership == i]
  if( length(unique(dat[clusterSet,1])) == 1) { # one label in this cluster
    clusterKSVM[[i]] = dat[clusterSet[1],1]
  } else {
    task2 = makeClassifTask(data = dat, target = "response")
    learner2 = makeLearner("classif.ksvm", par.vals = list(kernel = "rbfdot", C = C, sigma = sigma))
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

ksvmPred = character(length(test.set))

for(i in 1:n) {
  testClusterSet = test.set[clusterPred == i]
  if(length(clusterKSVM[[i]]) == 1) { # if cluster has one label
    ksvmPred[clusterPred == i] = as.character(clusterKSVM[[i]])
  } else {
    tmpPred = predict(clusterKSVM[[i]], task = task2, subset = testClusterSet)
    ksvmPred[clusterPred == i] =  as.character(tmpPred$data$response)
  }
}

#------------------------#
# prediction results     #
#------------------------#
cat("# Show prediction results:")
ksvmPred = as.factor(ksvmPred)
print(table(ksvmPred, dat[test.set,1]))

print(sum(diag(table(ksvmPred, dat[test.set,1])))/length(test.set)
