
library(nnet)
library(caret)

########################
# load all processed data
########################
# train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
# train_base1  <- read.csv('../processed_data/train_baselineNEW.csv', header = TRUE)
train <- read.csv('../processed_data/train_baselineCLEAN.csv', header = TRUE)
# train_des_st = read.csv('../../processed_data/train_description_sentiment.csv', header = TRUE)
# train_des_tfidf = read.csv('../../processed_data/train_description_wordcount_tfidf.csv', header = TRUE)


# train = merge(train_base, train_base1)
# train = merge(train, train_des_st)
# train = merge(train, train_des_tfidf)

########################
# define the categorical and numeric features
########################
# Check the type of each variable
sapply(train, class)

# sapply(train_des_tfidf, class)
# sapply(train_des_st, class)

numVars = c('bedrooms', 'bathrooms', 'price','numFeat', 'numPh', 'distance_city', 'clean_wordcount',
            'cats', '')
# ,
#             names(train_des_st)[!names(train_des_st) %in% c("listing_id", "interest_level", "interest_Nbr")],
#             names(train_des_tfidf)[!names(train_des_tfidf) %in% c("listing_id", "interest_level", "interest_Nbr")]
# )

catVars = c('weekend', 'created_month', 'created_hour','time_ofday')
for(var in catVars){
  train[var] = lapply(train[var], factor)
}

Xfeat <- names(train)[c(-1,-8, -9)]

targetVar <- 'interest_level'

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

## adding time_ofday to the model 
# 
# time.tag <-  c("00", "06", "12", "18", "24")
# 
# train$time_ofday <- cut(as.numeric(train$created_hour), breaks=time.tag,
#                         labels = c("00-06","06-12", "12-18", "18-23"), include.lowest= TRUE)
# 
# train$time_ofday <- as.factor(train$time_ofday)

##
inTrain <- createDataPartition(y = train$interest_level_num, list = FALSE, p = 0.8)
train.train <- train[inTrain,]
train.test <- train[-inTrain,]

# modelForm = createModelFormula(targetVar, c(catVars, numVars), FALSE)
modelForm = createModelFormula(targetVar, Xfeat)


###################
# Random Forest   #
###################

library(randomForest)


fit <-  randomForest(modelForm, data = train.train, importance = TRUE, ntree = 150, parallel =TRUE )

plot(fit)
varImpPlot(fit)

pred1 <- predict(fit, train.test)
compare = as.data.frame(cbind(pred1,train.test$interest_level))
colnames(compare) = c('predicted', 'real')
conf_matrix <- table(compare$predicted, compare$real)
confusionMatrix(conf_matrix)

########### Result with DD's Features #################
# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  168   36  104
# 2  336 6409 1624
# 3  263  329  600
# 
# Overall Statistics
# 
# Accuracy : 0.7272         
# 95% CI : (0.7183, 0.736)
# No Information Rate : 0.6864         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.2952         
# Mcnemar's Test P-Value : < 2.2e-16      
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.21904   0.9461   0.2577
# Specificity           0.98462   0.3667   0.9215
# Pos Pred Value        0.54545   0.7658   0.5034
# Neg Pred Value        0.93735   0.7567   0.8009
# Prevalence            0.07772   0.6864   0.2359
# Detection Rate        0.01702   0.6494   0.0608
# Detection Prevalence  0.03121   0.8480   0.1208
# Balanced Accuracy     0.60183   0.6564   0.5896
########### Result with DD's Features #################

############### Result with clean_wordcount (CLEAN.csv)  ###################
# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  171   52  101
# 2  321 6385 1494
# 3  275  449  621
# 
# Overall Statistics
# 
# Accuracy : 0.7272         
# 95% CI : (0.7183, 0.736)
# No Information Rate : 0.6977         
# P-Value [Acc > NIR] : 6.451e-11      
# 
# Kappa : 0.2954         
# Mcnemar's Test P-Value : < 2.2e-16      
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.22295   0.9272  0.28023
# Specificity           0.98319   0.3916  0.90540
# Pos Pred Value        0.52778   0.7787  0.46171
# Neg Pred Value        0.93756   0.6998  0.81288
# Prevalence            0.07772   0.6977  0.22454
# Detection Rate        0.01733   0.6470  0.06292
# Detection Prevalence  0.03283   0.8309  0.13629
# Balanced Accuracy     0.60307   0.6594  0.59282

############### Result with clean_wordcount ###################

require(doMC)
require(caret)
doMC::registerDoMC(cores=3)

# modelForm1 <- createModelFormula(targetVar, c(catVars, numVars))

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
fit1 <-  randomForest(modelForm, data = train.train, importance = TRUE, 
                      ntree = 150, parallel =TRUE, trcontrol = control )

pred2 <- predict(fit1, train.test)
compare = as.data.frame(cbind(pred2,train.test$interest_level))
colnames(compare) = c('predicted', 'real')
conf_matrix <- table(compare$predicted, compare$real)
confusionMatrix(conf_matrix)

########### Tuned Result with DD's Features #################
# 
# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  175   36  111
# 2  335 6399 1615
# 3  257  339  602
# 
# Overall Statistics
# 
# Accuracy : 0.7271          
# 95% CI : (0.7182, 0.7359)
# No Information Rate : 0.6864          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.297           
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.22816   0.9446   0.2586
# Specificity           0.98385   0.3700   0.9210
# Pos Pred Value        0.54348   0.7664   0.5025
# Neg Pred Value        0.93799   0.7533   0.8009
# Prevalence            0.07772   0.6864   0.2359
# Detection Rate        0.01773   0.6484   0.0610
# Detection Prevalence  0.03263   0.8460   0.1214
# Balanced Accuracy     0.60601   0.6573   0.5898
########### End Tuned Result with DD's Features #################



############### Tuning after adding clean_wordcount  (CLEAN.csv) #############
# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  176   53  113
# 2  319 6385 1472
# 3  272  448  631
# 
# Overall Statistics
# 
# Accuracy : 0.7287          
# 95% CI : (0.7199, 0.7375)
# No Information Rate : 0.6977          
# P-Value [Acc > NIR] : 6.715e-12       
# 
# Kappa : 0.3018          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.22947   0.9272  0.28475
# Specificity           0.98176   0.3996  0.90592
# Pos Pred Value        0.51462   0.7809  0.46706
# Neg Pred Value        0.93797   0.7041  0.81392
# Prevalence            0.07772   0.6977  0.22454
# Detection Rate        0.01783   0.6470  0.06394
# Detection Prevalence  0.03465   0.8285  0.13689
# Balanced Accuracy     0.60561   0.6634  0.59533
################### Tuning after adding clean_wordcount  ############



################ without sentiment  (11_v3.csv) #######################
# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  188   52  114
# 2  305 6312 1438
# 3  274  543  640
# 
# Overall Statistics
# 
# Accuracy : 0.7237          
# 95% CI : (0.7148, 0.7325)
# No Information Rate : 0.7001          
# P-Value [Acc > NIR] : 1.324e-07       
# 
# Kappa : 0.2966          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.24511   0.9139  0.29197
# Specificity           0.98176   0.4109  0.89354
# Pos Pred Value        0.53107   0.7836  0.43926
# Neg Pred Value        0.93913   0.6715  0.81544
# Prevalence            0.07774   0.7001  0.22218
# Detection Rate        0.01906   0.6398  0.06487
# Detection Prevalence  0.03588   0.8164  0.14768
# Balanced Accuracy     0.61343   0.6624  0.59275
################ without sentiment #######################




#### Tuning with (11_v3.csv)    ################
# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  145   30   86
# 2  423 6526 1752
# 3  199  264  441
# 
# Overall Statistics
# 
# Accuracy : 0.7209          
# 95% CI : (0.7119, 0.7297)
# No Information Rate : 0.6913          
# P-Value [Acc > NIR] : 7.239e-11       
# 
# Kappa : 0.2397          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.18905   0.9569  0.19351
# Specificity           0.98725   0.2859  0.93897
# Pos Pred Value        0.55556   0.7500  0.48783
# Neg Pred Value        0.93524   0.7476  0.79491
# Prevalence            0.07774   0.6913  0.23100
# Detection Rate        0.01470   0.6615  0.04470
# Detection Prevalence  0.02645   0.8819  0.09163
# Balanced Accuracy     0.58815   0.6214  0.56624





#### Tuning with (11_v3.csv)    ################



##################  with sentiment (11_v3.csv) ############################

# Confusion Matrix and Statistics
# 
# 
# 1    2    3
# 1  152   30   83
# 2  429 6507 1785
# 3  186  261  433
# 
# Overall Statistics
# 
# Accuracy : 0.7188          
# 95% CI : (0.7098, 0.7277)
# No Information Rate : 0.689           
# P-Value [Acc > NIR] : 5.914e-11       
# 
# Kappa : 0.236           
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.19817   0.9572  0.18818
# Specificity           0.98758   0.2784  0.94091
# Pos Pred Value        0.57358   0.7461  0.49205
# Neg Pred Value        0.93594   0.7459  0.79212
# Prevalence            0.07774   0.6890  0.23323
# Detection Rate        0.01541   0.6595  0.04389
# Detection Prevalence  0.02686   0.8839  0.08920
# Balanced Accuracy     0.59288   0.6178  0.56455
##################  with sentiment ############################


