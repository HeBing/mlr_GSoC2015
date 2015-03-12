n#----------------------------
# 2015-03-12 BH
# reproduce results for section 4
# for GSoC 2015
# mlr project: ensemble SVM
# ref: http://jmlr.org/proceedings/papers/v31/gu13b.html
#----------------------------

source("./CSVM.R")

#------------------------#
# synthetic dataset:     #
#   four Gaussian,       #
#   Section 4.1 in paper #
#------------------------#

# simulate data: 4 Gaussian
set.seed(20150312)
dat = rbind( mvrnorm(100, mu = c(-2,0), Sigma = diag(rep(0.1,2))),
              mvrnorm(100, mu = c(2,0), Sigma = diag(rep(0.1,2))),
              mvrnorm(100, mu = c(0,-2), Sigma = diag(rep(0.1,2))),
              mvrnorm(100, mu = c(0,2), Sigma = diag(rep(0.1,2))) )
dat = data.frame(dat)
dat = cbind(label = gl(2,200),dat)

training.set = sample(1:nrow(dat), round(nrow(dat)/2))
test.set = setdiff(1:nrow(dat), training.set)

# set parameter
k = 2
C = 5
lambda = 5
feature = dat[training.set,-1]
label = dat[training.set,1]

# CSVM train
myTrainedModel = trainCSVM(k = k, C = C, lambda = lambda, label = label, feature = feature)

# CSVM predict
myCSVMpred = predictCSVM(newdata = dat[test.set,-1], trainedKSVM = myTrainedModel)

# reproduce Figure 1 in paper, panel 3
plot(dat[test.set,-1], col = as.numeric(myCSVMpred), xlab="feature 1", ylab="feature 2")
legend("topright", legend = unique(myCSVMpred), col = as.numeric(unique(myCSVMpred)), pch=1)


#------------------------#
# real dataset:
#   data/svmguide1.tr
#	test/svmguide1.t
# ref: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#svmguide1
#------------------------#

## read in train data
dat = read.table("./data/svmguide1.tr", stringsAsFactor = F)
dat[dat[,1] == 0,1] = -1
dat[,1] = as.factor(as.character(dat[,1]))
for(i in 2:ncol(dat)) {
  dat[,i] = as.numeric(unlist(strsplit(dat[,i], split = ":"))[(1:(nrow(dat)*2)) %% 2 == 0])
}
colnames(dat)[1] = "response"
trainDat = dat

## read in train data
dat = read.table("./data/svmguide1.t", stringsAsFactor = F)
dat[dat[,1] == 0,1] = -1
dat[,1] = as.factor(as.character(dat[,1]))
for(i in 2:ncol(dat)) {
  dat[,i] = as.numeric(unlist(strsplit(dat[,i], split = ":"))[(1:(nrow(dat)*2)) %% 2 == 0])
}
colnames(dat)[1] = "response"
testDat = dat

# set parameter
k = 8
C = 100 ## {1,5,10,20,50,100}
lambda = 10 ## {1,5,10,20,50,100}
feature = trainDat[,-1]
label = trainDat[,1]

# CSVM train
myTrainedModel = trainCSVM(k = k, C = C, lambda = lambda, label = label, feature = feature)

# CSVM predict
myCSVMpred = predictCSVM(newdata = testDat[,-1], trainedKSVM = myTrainedModel)

# reproduce Figure 2
table(myCSVMpred, testDat$response)



#------------------------#
# real dataset:
#   data/ijcnn1.tr
#	test/ijcnn1.t
# ref: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1
#------------------------#

## read in train data
dat = read.table("./data/ijcnn1.tr", stringsAsFactor = F)
dat[dat[,1] == 0,1] = -1
dat[,1] = as.factor(as.character(dat[,1]))
for(i in 2:ncol(dat)) {
  dat[,i] = as.numeric(unlist(strsplit(dat[,i], split = ":"))[(1:(nrow(dat)*2)) %% 2 == 0])
}
colnames(dat)[1] = "response"
trainDat = dat

## read in train data
dat = read.table("./data/ijcnn1.t", stringsAsFactor = F)
dat[dat[,1] == 0,1] = -1
dat[,1] = as.factor(as.character(dat[,1]))
for(i in 2:ncol(dat)) {
  dat[,i] = as.numeric(unlist(strsplit(dat[,i], split = ":"))[(1:(nrow(dat)*2)) %% 2 == 0])
}
colnames(dat)[1] = "response"
testDat = dat

# set parameter
k = 8
C = 100 ## {1,5,10,20,50,100}
lambda = 10 ## {1,5,10,20,50,100}
feature = trainDat[,-1]
label = trainDat[,1]

# CSVM train
myTrainedModel = trainCSVM(k = k, C = C, lambda = lambda, label = label, feature = feature)

# CSVM predict
myCSVMpred = predictCSVM(newdata = testDat[,-1], trainedKSVM = myTrainedModel)

# reproduce Figure 2
table(myCSVMpred, testDat$response)

