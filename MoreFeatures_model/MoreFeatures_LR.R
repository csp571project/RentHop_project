
library(nnet)
library(caret)

########################
# load all processed data
########################
train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
train_des_st = read.csv('../processed_data/train_description_sentiment.csv', header = TRUE)
train_des_tfidf = read.csv('../processed_data/train_description_wordcount_tfidf.csv', header = TRUE)

train = merge(train_base, train_des_st)
train = merge(train, train_des_tfidf)

########################
# define the categorical and numeric features
########################
# Check the type of each variable
sapply(train_base, class)
sapply(train_des_tfidf, class)
sapply(train_des_st, class)

numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city',
            names(train_des_st)[!names(train_des_st) %in% c("listing_id", "interest_level", "interest_Nbr")],
            names(train_des_tfidf)[!names(train_des_tfidf) %in% c("listing_id", "interest_level", "interest_Nbr")]
            )

catVars = c('weekend', 'created_month')
for(var in catVars){
  train[var] = lapply(train[var], factor)
}

# Reset the interest level, set the 'low' as the base line of the model
train$interest_level_mult <- relevel(train$interest_level, ref = "low")
targetVar = 'interest_level_mult'

########################
# split into train and test
########################
# Create a model with multinom
createModelFormula <- function(targetVar, xVars, includeIntercept = TRUE){
  if(includeIntercept){
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ ')))
  } else {
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ '), -1))
  }
  return(modelForm)
}

# Split into train and test sets
inTrain <- createDataPartition(y = train$interest_level_num, list = FALSE, p = 0.8)
train.train <- train[inTrain,]
train.test <- train[-inTrain,]

# modelForm = createModelFormula(targetVar, c(catVars, numVars), FALSE)
modelForm = createModelFormula(targetVar, c(catVars, numVars))
model <- multinom(modelForm, data = train.train)
summary(model)

# Check the accuracy of the prediction
# predicted = predict(model, train.test, type = "probs")
# compare = as.data.frame(cbind(apply(predicted,1, which.max),train.test$interest_level_num))
predicted = predict(model, train.test)
compare = as.data.frame(cbind(predicted,train.test$interest_level_mult))
colnames(compare) = c('predicted', 'real')

conf_matrix <- table(compare$predicted, compare$real)
confusionMatrix(conf_matrix)

accuracy = mean(compare$predicted == compare$real)
accuracy

########################
# log-loss (the evaluation method used in kaggle)
########################
LogLoss = function(actual, predicted, eps = 1e-15) {
  for(i in 1:3){
    predicted[,i] = sapply(predicted[,i], function(v) pmin(pmax(v, eps), 1-eps))
  }
  return(-mean(sapply(1:dim(predicted)[1], function(i) log(predicted[i, as.character(actual[i])]))))
}

predicted = predict(model, train.test, type = "probs")
logloss = LogLoss(train.test$interest_level_mult, predicted)
logloss
# [1] 0.7398928

########################
# cross validation for accuracy
########################
#Perform 10 fold cross validation
cross_validation = function(data = data, targetVar = targetVar, xVars = xVars, includeIntercept = TRUE){
  accuracy = numeric()
  for(i in 1:10){
    if(i<10){
      fold = paste('Fold0', i, sep = '')
    }else{
      fold = paste('Fold', i, sep = '')
    }
    testIndexes <- createFolds(data$interest_level_mult, k=10, list=TRUE)[[fold]]
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    
    modelForm = createModelFormula(targetVar, xVars, includeIntercept)
    model <- multinom(modelForm, data = trainData)
    
    predicted = predict(model, testData)
    compare = as.data.frame(cbind(predicted, testData$interest_level_mult))
    colnames(compare) = c('predicted', 'real')
    accuracy = c(accuracy, mean(compare$predicted == compare$real))
  }
  return(accuracy)
}

accuracy = cross_validation(train, targetVar, c(catVars, numVars))
accuracy
# [1] 0.6869301 0.6857780 0.6934765 0.6855117 0.6824721 0.6914506 0.6966565 0.6893617 0.6893617
# [10] 0.6914506

mean(accuracy)
# [1] 0.6892449

# modeling without intercept
accuracy = cross_validation(train, targetVar, c(catVars, numVars), FALSE)
accuracy
# [1] 0.6900324 0.6888169 0.6812563 0.6876013 0.6883485 0.6855117 0.6869301 0.6994934 0.6907801
# [10] 0.6913880

mean(accuracy)
# [1] 0.6890159

# accuracy has no much difference with and without intercept

