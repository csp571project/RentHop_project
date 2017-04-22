
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

###################
# Random Forest   #
###################

library(randomForest)
fit <-  randomForest(modelForm, data = train.train, importance = TRUE, ntree = 250 )
varImpPlot(fit)

pred1 <- predict(fit, train.test)
compare = as.data.frame(cbind(pred1,train.test$interest_level_mult))
colnames(compare) = c('predicted', 'real')
conf_matrix <- table(compare$predicted, compare$real)
confusionMatrix(conf_matrix)
# Confusion Matrix and Statistics

# 
# 1    2    3
# 1 6496  413 1659
# 2   45  170   96
# 3  348  184  458
# 
# Overall Statistics
# 
# Accuracy : 0.7219          
# 95% CI : (0.7129, 0.7307)
# No Information Rate : 0.698           
# P-Value [Acc > NIR] : 1.108e-07       
# 
# Kappa : 0.2463          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: 1 Class: 2 Class: 3
# Sensitivity            0.9430  0.22164  0.20696
# Specificity            0.3047  0.98451  0.93051
# Pos Pred Value         0.7582  0.54662  0.46263
# Neg Pred Value         0.6979  0.93754  0.80234
# Prevalence             0.6980  0.07772  0.22424
# Detection Rate         0.6582  0.01723  0.04641
# Detection Prevalence   0.8682  0.03151  0.10031
# Balanced Accuracy      0.6238  0.60308  0.56874

accuracy = mean(compare$predicted == compare$real)
accuracy

