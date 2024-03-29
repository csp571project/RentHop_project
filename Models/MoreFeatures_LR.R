
library(nnet)
library(caret)

########################
# load all processed data
########################
train_base <- read.csv('../processed_data/train_moreFeat36.csv', header = TRUE)
train_des_st = read.csv('../processed_data/train_description_sentiment.csv', header = TRUE)
train_des_tfidf = read.csv('../processed_data/train_description_tfidf.csv', header = TRUE)

train1 = merge(train_base, train_des_st)
train2 = merge(train1, train_des_tfidf)

train=train2
########################
# define the categorical and numeric features
########################
# Check the type of each variable
# sapply(train_base, class)
# sapply(train_des_tfidf, class)
# sapply(train_des_st, class)

numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city', 'clean_wordcount','time_ofday',
            "cats", "dishwasher","dogs","doorman", "elevator", 
            "fee", "fitness", "hardwoods","laundry", "war",
            "room_sum", "room_diff", "room_price", "bed_ratio",
            "building_id", "display_address", "manager_id", "street_address"
            , names(train_des_st)[!names(train_des_st) %in% c("listing_id", "interest_level", "interest_Nbr")]
            , names(train_des_tfidf)[!names(train_des_tfidf) %in% c("listing_id", "interest_level", "interest_Nbr")]
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
inTrain <- createDataPartition(y = train$interest_level, list = FALSE, p = 0.8)
train.train <- train[inTrain,]
train.test <- train[-inTrain,]

# modelForm = createModelFormula(targetVar, c(catVars, numVars), FALSE)
modelForm = createModelFormula(targetVar, c(catVars, numVars))
model <- multinom(modelForm, data = train.train)
#summary(model)

# AIC without tfidf
# [1] 52260.92
# AIC with tfidf
# [1] 54557.11 


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
#[1] 0.7025456

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

mean(accuracy)
# old Baseline + senti only
# [1] 0.6939069
# old Baseline + senti + old tfidf table
# [1] 0.6892449
# old Baseline + senti + new tfidf table
# [1] 0.6911437

# new Baseline with scores + senti + old tfidf
# [1] 0.7746687

# new Baseline without scores + senti only
# [1] 0.6954699
# new Baseline without scores + senti + old tfidf table
# [1] 0.695843
# new Baseline without scores + senti + new tfidf table
# [1] 0.6984046

# new Baseline with features + senti only
# [1] 0.6991188
# new Baseline with features + senti + new tfidf table
# [1] 0.7008571


# modeling without intercept (It usually gets worse with logistic regression)
accuracy = cross_validation(train, targetVar, c(catVars, numVars), FALSE)
accuracy

mean(accuracy)
# old Baseline + senti only
# [1] 0.6935325
# old Baseline + senti + old tfidf table
# [1] 0.6890159
# old Baseline + senti + new tfidf table
# [1] 0.6907381

# new Baseline with scores + senti + old tfidf table
# [1] 0.770764

# new Baseline without scores + senti only
# [1] 0.6960982
# new Baseline without scores + senti + old tfidf table
# [1] 0.6960355
# new Baseline without scores + senti + new tfidf table
# [1] 0.6941117

# accuracy has no much difference with and without intercept

########################
# output test set probability
########################
train_base = read.csv('../processed_data/train_moreFeat36.csv', header = TRUE)
train_des_st = read.csv('../processed_data/train_description_sentiment.csv', header = TRUE)
train_des_tfidf = read.csv('../processed_data/train_description_tfidf.csv', header = TRUE)

train = merge(train_base, train_des_st)
train = merge(train, train_des_tfidf)

numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city', 'clean_wordcount','time_ofday',
            "cats", "dishwasher","dogs","doorman", "elevator", 
            "fee", "fitness", "hardwoods","laundry", "war",
            "room_sum", "room_diff", "room_price", "bed_ratio"
            , names(train_des_st)[!names(train_des_st) %in% c("listing_id", "interest_level", "interest_Nbr")]
            , names(train_des_tfidf)[!names(train_des_tfidf) %in% c("listing_id", "interest_level", "interest_Nbr")]
)

catVars = c('weekend', 'created_month', 'created_hour', "building_id", "display_address", "manager_id", "street_address")
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
test_des_st = read.csv('../processed_data/test_description_sentiment.csv', header = TRUE)
test_des_tfidf = read.csv('../processed_data/test_description_tfidf.csv', header = TRUE)

test = merge(test_base, test_des_st)
test = merge(test, test_des_tfidf)


for(var in catVars){
  test[var] = lapply(test[var], factor)
}

predicted = predict(model, test, type = 'prob')
#predicted = predict(model, test, type = 'class')
#table(predicted)

predicted = cbind(test$listing_id, predicted)
colnames(predicted)[1] = 'listing_id'

write.csv(predicted, file = '../processed_data/test_result_baseline_LR_new.csv', row.names = FALSE)





