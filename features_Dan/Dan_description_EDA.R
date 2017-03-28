setwd("/Users/Dan/2017 spring/CSP 571/project/data")
transfer=list('low'=1, 'medium'=2, 'high'=3)

####################
# Read description directly from previously processed csv
# Modeling with word count
description = read.csv(file='train_description.csv', header=TRUE, stringsAsFactors = FALSE)
non_empty_clean_des = description[which(description$clean_wordcount!=0), ]

library('caret')
inTrain <- createDataPartition(y = non_empty_clean_des[,'interest_Nbr'], list = FALSE, p = .8)
description_train <- non_empty_clean_des[inTrain,]
description_test <- non_empty_clean_des[-inTrain,]
stopifnot(nrow(description_train) + nrow(description_test) == nrow(description))

library("nnet")
description_model=multinom(interest_level~clean_wordcount, data=non_empty_clean_des)

summary(description_model)
# Call:
#   multinom(formula = interest_level ~ clean_wordcount, data = non_empty_clean_des)
# 
# Coefficients:
#   (Intercept) clean_wordcount
# low      2.0056566     0.001504118
# medium   0.8798079     0.002782457
# 
# Std. Errors:
#   (Intercept) clean_wordcount
# low     0.03454567    0.0004443611
# medium  0.03777511    0.0004800372
# 
# Residual Deviance: 74153.25 
# AIC: 74161.25 

description_fitted.results <- predict(description_model,newdata = description_test,"probs")

# Show the probability distribution over three interest levels
temp <- data.frame(interest_level = factor(rep(colnames(description_fitted.results), 
                                               each=dim(description_fitted.results)[1])), 
                   prob = c(as.vector(description_fitted.results[,1]), 
                            as.vector(description_fitted.results[,2]), 
                            as.vector(description_fitted.results[,3])))
ggplot(temp, aes(x=prob, colour=interest_level)) +
  geom_density() +
  xlab("Predicted probability") 

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

description_test[,'predicted_class'] = predictMNL(description_model,description_test)

table(description_test$predicted_class)
#low medium high
# 1    2    3 
# 6172 2152  812 

# use a confusion matrix to evaluate how good our results are
library('e1071')
confusion <- confusionMatrix(data = description_test$predicted_class
                             , reference = description_test$interest_Nbr
                             , dnn = c("Predicted level", 'Actual level')
)
confusion
# Confusion Matrix and Statistics
# 
# Actual level
# Predicted level    1    2    3
# 1 4194 1476  502
# 2 1465  502  185
# 3  549  199   64
# 
# Overall Statistics
# 
# Accuracy : 0.521           
# 95% CI : (0.5107, 0.5313)
# No Information Rate : 0.6795          
# P-Value [Acc > NIR] : 1.0000          
# 
# Kappa : -0.0031         
# Mcnemar's Test P-Value : 0.4482          
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity            0.6756  0.23059 0.085220
# Specificity            0.3245  0.76290 0.910793
# Pos Pred Value         0.6795  0.23327 0.078818
# Neg Pred Value         0.3205  0.76017 0.917468
# Prevalence             0.6795  0.23829 0.082202
# Detection Rate         0.4591  0.05495 0.007005
# Detection Prevalence   0.6756  0.23555 0.088879
# Balanced Accuracy      0.5000  0.49674 0.498006

library('pROC')
multiclass.roc(
  response=description_test$interest_Nbr,
  predictor=description_test$predicted_class)
# Multi-class area under the curve: 0.501
