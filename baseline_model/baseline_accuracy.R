
library(nnet)
library(caret)

train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
train_base1 <- read.csv('../processed_data/train_baselineNEW.csv', header =)

train = merge(train_base, train_base1)

########################
# define the categorical and numeric features
########################
# Check the type of each variable
sapply(train_base, class)
sapply(train_base1, class)

base_numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city'
                 #'manager_score','building_score'
                 )
base_catVars = c('weekend', 'created_month', 'created_hour')
for(var in base_catVars){
  train[var] = lapply(train[var], factor)
}

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

# modelForm = createModelFormula(targetVar, c(base_catVars, base_numVars), TRUE)
modelForm = createModelFormula(targetVar, c(base_catVars, base_numVars))
base_model <- multinom(modelForm, data = train.train)

# Check the accuracy of the prediction
# predicted = predict(base_model, train_base.test, type = "probs")
# compare = as.data.frame(cbind(apply(predicted,1, which.max),train.test$interest_level_num))
predicted = predict(base_model, train.test)
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

predicted = predict(base_model, train_base.test, type = "probs")
logloss = LogLoss(train_base.test$interest_level_mult, predicted)
logloss
# old Baseline
# [1] 0.7136462
# new Baseline
# [1] 0.509708

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

accuracy = cross_validation(train, targetVar, c(base_catVars, base_numVars))
accuracy
# old Baseline
# [1] 0.6884880 0.6844345 0.6881459 0.6873354 0.6891591 0.6865248 0.6936170 0.6904376 0.6901722
# [10] 0.6952381
# new Baseline with scores
# [1] 0.7777102 0.7797366 0.7870314 0.7819656 0.7842382 0.7734549 0.7816045
# [8] 0.7876393 0.7852512 0.7816744

mean(accuracy)
# old Baseline
# [1] 0.6893553
# new Baseline with scores
# [1] 0.7820306

# modeling without intercept
accuracy = cross_validation(train, targetVar, c(base_catVars, base_numVars), FALSE)
accuracy
# old Baseline
# [1] 0.6934144 0.6974671 0.6911854 0.6879433 0.6881459 0.6869301 0.6876013 0.6877406 0.6873354
# [10] 0.6882091
# new Baseline with scores
# [1] 0.7742655 0.7852077 0.7768997 0.7872340 0.7827761 0.7814019 0.7825735
# [8] 0.7866261 0.7856130 0.7839919

mean(accuracy)
# old Baseline
# [1] 0.6895972
# new Baseline with scores
# [1] 0.7826589

# accuracy has no much difference with and without intercept

########################
# output test set probability
########################
train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
train_base1  <- read.csv('../processed_data/train_baselineNEW.csv', header = TRUE)
train = merge(train_base, train_base1)

numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city'
)

catVars = c('weekend', 'created_month', 'created_hour')
for(var in catVars){
  train[var] = lapply(train[var], factor)
}

# Reset the interest level, set the 'low' as the base line of the model
train$interest_level_mult <- relevel(train$interest_level, ref = "low")
targetVar = 'interest_level'

# Create a model with multinom
createModelFormula <- function(targetVar, xVars, includeIntercept = TRUE){
  if(includeIntercept){
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ ')))
  } else {
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ '), -1))
  }
  return(modelForm)
}
modelForm = createModelFormula(targetVar, c(catVars, numVars))
model <- multinom(modelForm, data = train)

test_base <- read.csv('../processed_data/test_baseline11_v1.csv', header = TRUE)
test = test_base

sapply(test, class)

for(var in catVars){
  test[var] = lapply(test[var], factor)
}

predicted = predict(model, test, type = 'prob')
#predicted = predict(model, test, type = 'class')
#table(predicted)

predicted = cbind(test$listing_id, predicted)
colnames(predicted)[1] = 'listing_id'

write.csv(predicted, file = '../processed_data/test_result_baseline_LR.csv', row.names = FALSE)

