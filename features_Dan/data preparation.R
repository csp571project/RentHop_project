setwd("/Users/Dan/2017 spring/CSP 571/project/data")
transfer=list('low'=1, 'medium'=2, 'high'=3)

####################
# Read from train data set
# library("rjson")
# files = list.files(pattern = '*.json')
# train_file<-files[2]
# train_data<-fromJSON(file=train_file)
# train_data <- lapply(train_data, function(x) {
#   x[sapply(x, is.null)] <- NA
#   unlist(x,use.names = FALSE)
# })

####################
# Construct price data frame
# price=data.frame(cbind(train_data$price, 
#                        train_data$interest_level),
#                  stringsAsFactors = FALSE)
# colnames(price)=c('price', 'interest_level')
# price$price=as.numeric(price$price)
# price$interest_Nbr <- unlist(sapply(price$interest_level, 
#                                     function(v) transfer[v]),
#                              use.names = FALSE)
# 
# write.csv(price, file= "train_price.csv", row.names=FALSE)

####################
# Read price directly from previously processed csv
price = read.csv(file = 'train_price.csv', header=TRUE, stringsAsFactors = FALSE)

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
table(price_test$predicted_class)
#low medium high
#1    2    3 
#6801 2268  800

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


#####################
# Construct the basic description data frame
# description=data.frame(cbind(train_data$listing_id,
#                             train_data$description, 
#                             train_data$interest_level),
#                       stringsAsFactors=FALSE)
# colnames(description) = c('listing_id', 'origin_des', 'interest_level')
# description$interest_Nbr <- unlist(sapply(description$interest_level, 
#                                             function(v) transfer[v]),
#                                    use.names = FALSE)
# 
# ## plain description is to remove the suffix
# # Get rid of the html patterns inside the text
# # https://tonybreyal.wordpress.com/2011/11/18/htmltotext-extracting-text-from-html-via-xpath/
# pattern = "</?\\w+((\\s+\\w+(\\s*=\\s*(?:\".*?\"|'.*?'|[^'\">\\s]+))?)+\\s*|\\s*)/?>"
# description$plain_des = gsub(pattern, "\\1", description$origin_des)
# 
# # Get rid of the ending word
# description$plain_des = gsub("website_redacted", " ", description$plain_des)
# description$plain_des = gsub("<a", " ", description$plain_des)
# 
# # This wordcount function does not distinguish duplicated words and will reserve single char and digit 
# wordcount = function(v){
#   temp = strsplit(v, "\\W+")[[1]]
#   if(sum(which(temp==""))>0)
#     temp = temp[-which(temp=="")]
#   return(length(temp))
# }
# description$plain_wordcount = vapply(description$plain_des, wordcount, integer(1), USE.NAMES = FALSE)
# 
# ## clean description is to remove the stopwords and punctuation
# #install.packages("tm", dependencies = TRUE)
# library('tm')
# temp = VCorpus(VectorSource(description$plain_des))
# 
# # Transfer to lower case
# corpus = tm_map(temp, content_transformer(tolower))
# 
# # Get rid of punctuations
# corpus = tm_map(corpus, removeWords, stopwords("english"))
# (f <- content_transformer(function(x, pattern) gsub(pattern, " ", x)))
# #corpus = tm_map(corpus, f, "[[:digit:]]+")
# corpus = tm_map(corpus, f, "[[:punct:]]")
# 
# # Get rid of extra white spaces
# corpus = tm_map(corpus, stripWhitespace)
# 
# description$clean_des = unlist(lapply(corpus, as.character))
# description$clean_wordcount = vapply(description$clean_des, wordcount, integer(1), USE.NAMES=FALSE)
# 
# write.csv(description, file= "train_description.csv", row.names=FALSE)

####################
# text mining
description = read.csv(file='train_description.csv', header=TRUE, stringsAsFactors = FALSE)

library('tm')

non_empty_clean_des = description[which(description$clean_wordcount!=0), ]

corpus = VCorpus(VectorSource(non_empty_clean_des$clean_des))
dtm = DocumentTermMatrix(corpus, control = list(wordLengths=c(1,Inf)))
findFreqTerms(dtm, 10000)

tfidf = weightSMART(dtm, spec='npn')


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

one_vs_all_roc(description_train,description_test)
