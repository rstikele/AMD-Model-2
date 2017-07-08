library(quantmod)
library(XML)
library(Amelia)
library(TTR)
library(plotly)
library(ggplot2)
library(binhf)
library(PerformanceAnalytics)
library(MuMIn)
library(glmulti)
library(caTools)
library(neuralnet)
library(ggplot2)
library(boot)
library(plyr)
library(gtools)
library(h2o)

#Execute AMD 1 of 2.R before this

amd.bb = BBands( AMD[,c("AMD.High","AMD.Low","AMD.Close")] )
amd.macd = MACD(AMD[,4])
amd.rsi = RSI(AMD[,4])
amd.cci = CCI(AMD[,c("AMD.High","AMD.Low","AMD.Close")])
amd.adx = ADX(AMD[,c("AMD.High","AMD.Low","AMD.Close")])
amd.stoc = stoch(AMD[,c("AMD.High","AMD.Low","AMD.Close")])
amd.cmf = CMF(AMD[,c("AMD.High","AMD.Low","AMD.Close")], AMD[,5])
amd.vwap = VWAP(AMD[,4], AMD[,5])
amd.evwma = EVWMA(AMD[,4], AMD[,5])
amd.will = WPR(AMD[,c("AMD.High","AMD.Low","AMD.Close")])
amd.smi = SMI(AMD[,c("AMD.High","AMD.Low","AMD.Close")])
amd.zlema = ZLEMA(AMD[,4])
amd.percent.change = percent.change(AMD[,1], AMD[,4])
amd.cci.rsi = amd.cci/amd.rsi
colnames(amd.cci.rsi) = 'amd.cci.rsi'
amd.zlema.macd = amd.zlema/amd.macd[,1]
colnames(amd.zlema.macd) = 'amd.zlema.macd'
amd.zlema.signal = amd.zlema/amd.macd[,2]
colnames(amd.zlema.signal) = 'amd.zlema.signal'
amd.macd.signal = amd.macd[,2]/amd.macd[,1]
colnames(amd.macd.signal) = 'amd.macd.signal'
amd.smi.signal = amd.smi[,1]/amd.smi[,2]
colnames(amd.smi.signal) = 'amd.smi.signal'
amd.dx.fastk = amd.adx[,3]*amd.stoc[,1]
colnames(amd.dx.fastk) = 'amd.dx.fastk'
amd.fast.slow.adx = (((amd.stoc[,2]+amd.stoc[,3])/2) * (amd.adx[,4]))
colnames(amd.fast.slow.adx) = 'amd.fast.slow.adx'

model1 = glm(amd.percent.change + AMD[,4] ~ amd.cci + amd.cmf + amd.evwma + amd.rsi + amd.vwap + amd.will + amd.zlema, amd.cci.rsi, amd.dx.fastk,amd.fast.slow.adx, amd.macd.signal,amd.smi.signal,amd.zlema.macd,amd.zlema.signal)#Set all variables to compare
model1in = glmulti(model1, level = 1, crit = "aicc")
summary(model1in) 
weightable(model1in)



b1 = glm(amd.percent.change + AMD[, 4] ~ 1 + amd.cci + amd.cmf + amd.evwma + amd.rsi + amd.vwap + amd.will + amd.zlema)
b2 = glm( amd.percent.change + AMD[, 4] ~ 1 + amd.cci + amd.cmf + amd.evwma + amd.rsi + amd.will + amd.zlema)
b3 = glm(amd.percent.change + AMD[, 4] ~ 1 + amd.cci + amd.evwma + amd.rsi + amd.will + amd.zlema)
b4 = glm(amd.percent.change + AMD[, 4] ~ 1 + amd.cci + amd.evwma + amd.rsi + amd.vwap + amd.will + amd.zlema)
av4 = model.avg(b1,b2,b3,b4)
summary(ave4)
summary(b1)

########################################################################################################################################################################################################################################
data = merge.xts(amd.bb, amd.macd, amd.rsi, amd.cci, amd.adx, amd.stoc, amd.cmf, amd.vwap,amd.evwma,amd.will,amd.smi,amd.zlema,amd.cci.rsi, amd.dx.fastk,amd.fast.slow.adx, amd.macd.signal,amd.smi.signal,amd.zlema.macd,amd.zlema.signal)
data = data.frame(data)
data = cbind.data.frame(AMD[,4],data)
data = na.contiguous(data)
maxs = apply(data,2,max)
mins = apply(data,2,min)
#Scale normalizes, and returns a matrix, must be converted back to df
scaled.data= scale(data, center = mins,scale = maxs-mins)#Normalizes data
scaled.data = data.frame(scaled.data)

split = sample.split(data, SplitRatio  =0.7)
train = subset(data, split == T)
test = subset(data, split == F)

missmap(data, main ='Missing Map', col = c('yellow', 'black'), legend = F)

n = names(train)
n
f = AMD.Close ~ dn + up + pctB + macd + mavg + signal + EMA + cci + DIp + DIn + DX + ADX + fastK + fastD + slowD + amd.cmf + VWAP + amd.evwma  + SMI + signal.1 + amd.zlema + amd.cci.rsi + amd.dx.fastk + amd.fast.slow.adx + amd.macd.signal + amd.smi.signal + amd.zlema.macd + amd.zlema.signal
f
nn = neuralnet(f, data = train, hidden = c(21,13,8,5,3), linear.output = T)
plot(nn)

predicted.nn.values = compute(nn, test[1:28])
true.predictons = predicted.nn.values$net.result * (max(data$AMD.Close)-min(data$AMD.Close))+min(data$AMD.Close)
test.r = test$AMD.Close * (max(data$AMD.Close)-min(data$AMD.Close))+min(data$AMD.Close)
MSE.nn = sum((test.r - true.predictons)^2)/nrow(test)
MSE.nn
error.df = data.frame(test.r,true.predictons)
tail(error.df)

cv.error <- NULL
k <- 10

pbar <- create_progress_bar('text')
pbar$init(k)
x = 28:1
y = permutations(n=28,r=2,v=x,repeats.allowed = T) #Used for optimizing hidden layers i neural net

#For two layers: (6,6); (7,25); (7,28) are under 1% error; determined by permutation

for(i in 1:k){
  index <- sample(1:nrow(data),round(0.9*nrow(data)))
  train.cv <- scaled.data[index,]
  test.cv <- scaled.data[-index,]
  
  nn <- neuralnet(f,data=train.cv, hidden= c(6, 6) , rep = 10, linear.output=T)
  
  pr.nn <- compute(nn,test.cv[,1:28])
  pr.nn <- pr.nn$net.result*(max(data$AMD.Close)-min(data$AMD.Close))+min(data$AMD.Close)
  
  test.cv.r <- (test.cv$AMD.Close)*(max(data$AMD.Close)-min(data$AMD.Close))+min(data$AMD.Close)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  
  pbar$step()
}

mean(cv.error)
cv.error
big.nn.one.frame = data.frame(big.nn.one)
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN c(7,25) 10 rep',horizontal=TRUE)

error.nn.df = data.frame(test.cv.r,pr.nn)

ggplot(error.nn.df, aes(x = test.cv.r, y = pr.nn)) + geom_point() + stat_smooth()

######################################################################################################
# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
pkgs <- c("statmod","RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-vajda/3/R")

# Finally, let's load H2O and start up an H2O cluster
library(h2o)
h2o.init(nthreads = -1)
browseURL('http://localhost:54321/flow/index.html#')


#############################################################################################################################
######################################################################################################################################################################################################################################
library(ISLR)
library(e1071)
library(MASS)


spl = sample.split(amd.data[1:21], 0.7)
train = subset(amd.data[1:21], spl == TRUE)
test = subset(amd.data[1:21], spl == FALSE)
model = svm(Percent.Change~., data = train)#Do a train test split, review cost and gamma
summary(model)
tune.results = tune(svm,train.x = train[2:21], train.y = train[,1], kernal = 'radial', ranges = list(cost=c(0.1,0.5,1,2,3,5,7,10,20,30,50,75,100), gamma = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.5,1,2,3,5,10)))#Decrease variables to speed up
summary(tune.results)
tuned.svm = svm(Percent.Change~., data = train, kernal='radial', cost = 100, gamma=0.01)
summary(tuned.svm)
tuned.predicted.values = predict(tuned.svm,test[2:21])

svm.result = data.frame(tuned.predicted.values,test[,1])
percent.error = function(predicted, actual){abs((actual-predicted)/predicted)*100}
svm.error = percent.error(predicted = tuned.predicted.values, actual = test[,1])
svm.result = cbind(svm.error,svm.result)
longs = ifelse(svm.result$test...1. > 0 & svm.result$tuned.predicted.values > 0, 1, 0)
shorts = ifelse(svm.result$test...1. < 0 & svm.result$tuned.predicted.values < 0, 1, 0)
svm.result = cbind(longs,svm.result)
svm.result = cbind(shorts,svm.result)
View(svm.result)
mean(svm.result$longs)
mean(svm.result$shorts)#SVM better at predicting shorts, but still pretty bad
