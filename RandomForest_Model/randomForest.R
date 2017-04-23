
library(nnet)
library(caret)

########################
# load all processed data
########################
train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
train_base1  <- read.csv('../processed_data/train_baselineNEW.csv', header = TRUE)
# train_des_st = read.csv('../processed_data/train_description_sentiment.csv', header = TRUE)
# train_des_tfidf = read.csv('../processed_data/train_description_wordcount_tfidf.csv', header = TRUE)

train = merge(train_base, train_base1)
# train = merge(train, train_des_st)
# train = merge(train, train_des_tfidf)

########################
# define the categorical and numeric features
########################
# Check the type of each variable
sapply(train_base, class)

# sapply(train_des_tfidf, class)
# sapply(train_des_st, class)

numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city', 
            'manager_score', 'building_score')
#             names(train_des_st)[!names(train_des_st) %in% c("listing_id", "interest_level", "interest_Nbr")],
#             names(train_des_tfidf)[!names(train_des_tfidf) %in% c("listing_id", "interest_level", "interest_Nbr")]
# )

catVars = c('weekend', 'created_month', 'created_hour')
for(var in catVars){
  train[var] = lapply(train[var], factor)
}


targetVar = 'interest_level'

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


###################
# Random Forest   #
###################

library(randomForest)
fit <-  randomForest(modelForm, data = train.train, importance = TRUE, ntree = 250 )
varImpPlot(fit)

pred1 <- predict(fit, train.test)
compare = as.data.frame(cbind(pred1,train.test$interest_level))
colnames(compare) = c('predicted', 'real')
conf_matrix <- table(compare$predicted, compare$real)
confusionMatrix(conf_matrix)


# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  312   25  120
# 2  113 6217 1033
# 3  342  546 1161
# 
# Overall Statistics
# 
# Accuracy : 0.7792          
# 95% CI : (0.7709, 0.7874)
# No Information Rate : 0.6878          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4919          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: 1 Class: 2 Class: 3
# Sensitivity           0.40678   0.9159   0.5017
# Specificity           0.98407   0.6280   0.8825
# Pos Pred Value        0.68271   0.8444   0.5666
# Neg Pred Value        0.95166   0.7721   0.8526
# Prevalence            0.07772   0.6878   0.2345
# Detection Rate        0.03161   0.6300   0.1176
# Detection Prevalence  0.04631   0.7461   0.2076
# Balanced Accuracy     0.69542   0.7720   0.6921


########################
# cross validation for accuracy
########################

cross_validation = function(data = data, targetVar = targetVar, xVars = xVars, includeIntercept = TRUE){
  accuracy = numeric()
  for(i in 1:10){
    if(i<10){
      fold = paste('Fold0', i, sep = '')
    }else{
      fold = paste('Fold', i, sep = '')
    }
    testIndexes <- createFolds(data$interest_level, k=10, list=TRUE)[[fold]]
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    
    modelForm = createModelFormula(targetVar, xVars, includeIntercept)
    model <- randomForest(modelForm, data = trainData)
    
    predicted = predict(model, testData)
    compare = as.data.frame(cbind(predicted, testData$interest_level))
    colnames(compare) = c('predicted', 'real')
    accuracy = c(accuracy, mean(compare$predicted == compare$real))
    print(accuracy)
  }
  return(accuracy)
}

accuracy = cross_validation(train, targetVar, c(catVars, numVars))
accuracy

# [1] 0.7947315 0.7811994 0.7932712 0.7864235 0.7829787 0.7847588
# [7] 0.7857722 0.7929078 0.7824149 0.7921394

mean(accuracy)




