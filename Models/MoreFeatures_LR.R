
library(nnet)
library(caret)

########################
# load all processed data
########################
train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
train_base1  <- read.csv('../processed_data/train_baselineNEW.csv', header = TRUE)
train_des_st = read.csv('../processed_data/train_description_sentiment.csv', header = TRUE)
train_des_tfidf = read.csv('../processed_data/train_description_wordcount_tfidf.csv', header = TRUE)

train = merge(train_base, train_base1)
train = merge(train, train_des_st)
train = merge(train, train_des_tfidf)


########################
# define the categorical and numeric features
########################
# Check the type of each variable
sapply(train_base, class)
sapply(train_base1, class)
sapply(train_des_tfidf, class)
sapply(train_des_st, class)

numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city', 'manager_score',
            'building_score',
            names(train_des_st)[!names(train_des_st) %in% c("listing_id", "interest_level", "interest_Nbr")],
            names(train_des_tfidf)[!names(train_des_tfidf) %in% c("listing_id", "interest_level", "interest_Nbr")]
            )

catVars = c('weekend', 'created_month', 'created_hour')
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
# 
# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1 6420  141 1107
# 2   27  280  163
# 3  441  346  944
# 
# Overall Statistics
# 
# Accuracy : 0.7745          
# 95% CI : (0.7662, 0.7828)
# No Information Rate : 0.6979          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4563          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: 1 Class: 2 Class: 3
# Sensitivity            0.9321  0.36506  0.42638
# Specificity            0.5813  0.97913  0.89719
# Pos Pred Value         0.8372  0.59574  0.54535
# Neg Pred Value         0.7874  0.94819  0.84394
# Prevalence             0.6979  0.07772  0.22434
# Detection Rate         0.6505  0.02837  0.09565
# Detection Prevalence   0.7770  0.04762  0.17540
# Balanced Accuracy      0.7567  0.67209  0.66178

accuracy = mean(compare$predicted == compare$real)
accuracy
# 0.7745466

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
# [1] 0.530166

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
# old Baseline + senti only
# [1] 0.6887538 0.6904376 0.7023303 0.6918558 0.7013776 0.6913880 0.6945683
# [8] 0.6959870 0.6891591 0.6932118
# old Baseline + senti + old tfidf table
# [1] 0.6869301 0.6857780 0.6934765 0.6855117 0.6824721 0.6914506 0.6966565 0.6893617 0.6893617
# [10] 0.6914506
# old Baseline + senti + new tfidf table
# [1] 0.6959076 0.7021277 0.6950355 0.6832827 0.6950972 0.6853728 0.6901722
# [8] 0.6786872 0.6867275 0.6990272
# new Baseline + senti + old tfidf
# [1] 0.7760892 0.7710233 0.7787234 0.7759319 0.7791734 0.7687943
# [7] 0.7698541 0.7755267 0.7775076 0.7740628

mean(accuracy)
# old Baseline + senti only
# [1] 0.6939069
# old Baseline + senti + old tfidf table
# [1] 0.6892449
# old Baseline + senti + new tfidf table
# [1] 0.6911437
# new Baseline + senti + old tfidf
# [1] 0.7746687

# modeling without intercept
accuracy = cross_validation(train, targetVar, c(catVars, numVars), FALSE)
accuracy
# old Baseline + senti only
# [1] 0.6955024 0.6899696 0.6916532 0.6971232 0.6895643 0.6934144 0.6973258
# [8] 0.6928065 0.6955024 0.6924635
# old Baseline + senti + old tfidf table
# [1] 0.6900324 0.6888169 0.6812563 0.6876013 0.6883485 0.6855117 0.6869301 0.6994934 0.6907801
# [10] 0.6913880
# old Baseline + senti + new tfidf table
# [1] 0.6893617 0.6920583 0.6873354 0.6924635 0.6932118 0.6918558 0.6895643
# [8] 0.6909828 0.6917933 0.6887538
# new Baseline + senti + old tfidf
# [1] 0.7671733 0.7650993 0.7663627 0.7752786 0.7677812 0.7722391
# [7] 0.7720827 0.7779579 0.7773501 0.7663154

mean(accuracy)
# old Baseline + senti only
# [1] 0.6935325
# old Baseline + senti + old tfidf table
# [1] 0.6890159
# old Baseline + senti + new tfidf table
# [1] 0.6907381
# new Baseline + senti + old tfidf
# [1] 0.770764

# accuracy has no much difference with and without intercept

