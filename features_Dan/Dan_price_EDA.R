transfer=list('low'=1, 'medium'=2, 'high'=3)

####################
# # Read from train data set
library("rjson")
train_data<-fromJSON('../train.json')
train_data <- lapply(train_data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x,use.names = FALSE)
})
test_file<-files[1]
test_data<-fromJSON(file=test_file)
test_data <- lapply(test_data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x,use.names = FALSE)
})

####################
# Construct price data frame
price=data.frame(cbind(train_data$price,
                       train_data$interest_level),
                 stringsAsFactors = FALSE)
colnames(price)=c('price', 'interest_level')
price$price=as.numeric(price$price)
price$interest_Nbr <- unlist(sapply(price$interest_level,
                                    function(v) transfer[v]),
                             use.names = FALSE)


####################
#install.packages('caret')
library('caret')
inTrain <- createDataPartition(y = price[,'interest_Nbr'], list = FALSE, p = .8)
price_train <- price[inTrain,]
price_test <- price[-inTrain,]
stopifnot(nrow(price_train) + nrow(price_test) == nrow(price))


library("nnet")
price_model=multinom(interest_level~price, data=price)
# commed + shift + c to commend a block
# # weights:  9 (4 variable)
# initial  value 54218.713670 
# iter  10 value 37336.650452
# final  value 37336.639462 
# converged
summary(price_model)
#Call:
#  multinom(formula = interest_level ~ price, data = price)
#Coefficients:
#  (Intercept)        price
#low     -0.0859740 0.0007297733
#medium  -0.2712489 0.0004596248
#Std. Errors:
#  (Intercept)        price
#low    2.109881e-09 5.988441e-06
#medium 1.951706e-09 6.377154e-06
#Residual Deviance: 74673.28 
#AIC: 74681.28 

price_fitted.results <- predict(price_model,newdata = price_test,"probs")

# Show the probability distribution over three interest levels
temp <- data.frame(interest_level = factor(rep(colnames(price_fitted.results), 
                                     each=dim(price_fitted.results)[1])), 
                  prob = c(as.vector(price_fitted.results[,1]), 
                             as.vector(price_fitted.results[,2]), 
                             as.vector(price_fitted.results[,3])))
ggplot(temp, aes(x=prob, colour=interest_level)) +
  geom_density() +
  xlab("Predicted probability") 


# If directly use predict(price_model,newdata = price_test,"class"), almost all data will
# be predicted as 'low'
# Each entry in price_fitted.results gives the probability of selecting a given option
# Therefore to make a choice, we need to calculate the cumulative probabilities associated 
# with each option. We can then draw a random value between 0 to 1; the option with the 
# greatest cumulative probability below our draw value is our choice. This can be written
# into a function for easier use.
# Reference:
# https://www.r-bloggers.com/how-to-multinomial-regression-models-in-r/
predictMNL <- function(model, newdata) {
  # Only works for neural network models
  if (is.element("nnet",class(model))) {
    # Calculate the individual and cumulative probabilities
    probs <- predict(model,newdata,"probs")
    cum.probs <- t(apply(probs,1,cumsum))
    
    # Draw random values
    vals <- runif(nrow(newdata))
    
    # Join cumulative probabilities and random draws
    tmp <- cbind(cum.probs,vals)

    # For each row, get choice index.
    k <- ncol(probs)
    ids <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
    tags = unlist(sapply(ids,function(v) transfer[colnames(probs)[v]]), use.names = FALSE)
    return(tags)
  }
}

price_test[,'predicted_class'] = predictMNL(price_model,price_test)
table(price_test$predicted_class, price_test$interest_Nbr)
#low medium high
#      1    2    3
# 1 4885 1464  473
# 2 1492  559  209
# 3  483  219   85

# use a confusion matrix to evaluate how good our results are
#install.packages('e1071')
library('e1071')
confusion <- confusionMatrix(data = price_test$predicted_class
                             , reference = price_test$interest_Nbr
                             , dnn = c("Predicted level", 'Actual level')
)
confusion
# Confusion Matrix and Statistics
# 
# Actual level
# Predicted level    1    2    3
# 1 6355 1959  629
# 2 1935  633  251
# 3  641  277  123
# 
# Overall Statistics
# 
# Accuracy : 0.5554         
# 95% CI : (0.5468, 0.564)
# No Information Rate : 0.6976         
# P-Value [Acc > NIR] : 1.0000         
# 
# Kappa : 0.0272         
# Mcnemar's Test P-Value : 0.6727         
# 
# Statistics by Class:
# 
#                      Class: 1 Class: 2 Class: 3
# Sensitivity            0.7116  0.22063 0.122632
# Specificity            0.3316  0.77995 0.922203
# Pos Pred Value         0.7106  0.22455 0.118156
# Neg Pred Value         0.3326  0.77604 0.925183
# Prevalence             0.6976  0.22409 0.078341
# Detection Rate         0.4964  0.04944 0.009607
# Detection Prevalence   0.6985  0.22018 0.081309
# Balanced Accuracy      0.5216  0.50029 0.522418


#install.packages('pROC')
library('pROC')
multiclass.roc(
  response=price_test$interest_Nbr,
  predictor=price_test$predicted_class)


# Since this is a multinomial case, I can't plot the roc curve directly. So I choose to use
# the one-vs-all classification with naivebayes model as following to plot the roc curve for
# each interest level
# Reference:
# http://stats.stackexchange.com/questions/71700/how-to-draw-roc-curve-with-three-response-variable/110550#110550
library(ROCR)
library(klaR)
one_vs_all_roc<-function(test, train, type_num=3){
  aucs = c()
  plot(x=NA, y=NA, xlim=c(0,1), ylim=c(0,1),
       ylab='True Positive Rate',
       xlab='False Positive Rate',
       bty='n')
  
  for (type.id in 1:type_num) {
    type = as.factor(train$interest_Nbr == type.id)
    
    nbmodel = NaiveBayes(type ~ price, data=train)
    nbprediction = predict(nbmodel, test,type='raw')
    
    score = nbprediction$posterior[, 'TRUE']
    actual.class = test$interest_Nbr == type.id
    
    pred = prediction(score, actual.class)
    nbperf = performance(pred, "tpr", "fpr")
    
    roc.x = unlist(nbperf@x.values)
    roc.y = unlist(nbperf@y.values)
    lines(roc.y ~ roc.x, col=type.id+1, lwd=2)
    
    nbauc = performance(pred, "auc")
    nbauc = unlist(slot(nbauc, "y.values"))
    aucs[type.id] = nbauc
  }
  lines(x=c(0,1), c(0,1))
  legend(0.8,0.3,legend=names(transfer), col=2:(1+type_num), lwd=2,text.width=0.12, 
         y.intersp = 1.5, title='interest level')
  return(mean(aucs))
}
one_vs_all_roc(price_train,price_test)
