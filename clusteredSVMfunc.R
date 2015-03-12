#----------------------------
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

# plot result
plot(dat[test.set,-1], col = as.numeric(myCSVMpred), xlab="feature 1", ylab="feature 2")
legend("topright", legend = unique(myCSVMpred), col = as.numeric(unique(myCSVMpred)), pch=1)

#------------------------#
# real dataset:
#   
