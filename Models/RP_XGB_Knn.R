library(lubridate)
library(dplyr)
library(jsonlite)
library(caret)
library(purrr)
library(xgboost)
library(MLmetrics)
library(tidytext)
library(reshape2)
seed = 11080
set.seed(seed)

train <- read.csv("../processed_data/train_baselineCLEAN.csv", header = TRUE)

featExtract <- function(c, lvl) {
  ###lvl controls what number of count to be considered as factors for numerics
  if (missing(lvl)) {
    lvl <- 10
  }
  fln <- unique(c)
  # if (is.numeric(c)) {
  ifelse(
    length(fln) == 2 & sum(unique(as.numeric(c))) == 1 ,
    "bin" ,
    ifelse(length(fln) >= 3 &
             length(unique(fln)) < lvl , "factor" , "num")
  )
}


binFeat <-
  names(which(lapply(train, featExtract)  == "bin"))
numFeat <-
  names(which(lapply(train, featExtract)  == "num"))
catFeat <-
  names(which(lapply(train, featExtract)  == "factor"))
strFeat <-
  names(which(lapply(train, featExtract)  == "char"))

train[numFeat] <- lapply(train[numFeat], as.numeric)
train[catFeat] <- lapply(train[catFeat], as.factor)
train[binFeat] <- lapply(train[binFeat], as.factor)

str(train)

## EXTRA
##
##train$bed_price <- train$price/train_test$bedrooms
##train[which(is.infinite(train$bed_price)),]$bed_price = 
##  train[which(is.infinite(train$bed_price)),]$price
##train_test$photo_count <- log(train_test$photo_count + 1)
##train_test$feature_count <- log(train_test$feature_count + 1)
##train_test$price <- log(train_test$price + 1)
##train_test$room_price <- log(train_test$room_price + 1)
##train_test$bed_price <- log(train_test$bed_price + 1)
##
#Convert labels to integers
require(doMC)
require(caret)
doMC::registerDoMC(cores=3)


train$interest_level<-as.integer(factor(train$interest_level))

train$week<-as.integer(factor(train$weekend))

xFeat <- names(train)[c(-1,-8, -9)]
yFeat <- 'interest_level_num'

createModelFormula <- function(targetVar, xVars, includeIntercept = TRUE){
  if(includeIntercept){
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ ')))
  } else {
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ '), -1))
  }
  return(modelForm)
}

modelForm <- createModelFormula(yFeat, xFeat)




inTrain <- createDataPartition(y = train$interest_level, list = FALSE, p = 0.95)
train.train <- train[inTrain,]
train.test <- train[-inTrain,]

y <- as.numeric(train.train$interest_level)

y = y - 1
train.train$interest_level = NULL
train.train$interest_level_num = NULL
#train.test$interest_level = NULL

###

##################
#Parameters for XGB

xgb_params = list(
  colsample_bytree= 0.7,
  subsample = 0.7,
  eta = 0.1,
  objective= 'multi:softprob',
  max_depth= 4,
  min_child_weight= 1,
  eval_metric= "mlogloss",
  num_class = 3,
  seed = seed
)
#############################################################

#convert xgbmatrix
train.trainM <- xgb.DMatrix(data.matrix(train.train))

#create folds
kfolds<- 10
folds<-createFolds(y, k = kfolds, list = TRUE, returnTrain = FALSE)
fold <- folds$Fold01



x_train<-train.train[-fold,] #Train set
x_val<-train.train[fold,] #Out of fold validation set


y_train<-y[-fold]
y_val<-y[fold]

#convert to xgbmatrix
dtrain <- xgb.DMatrix(data.matrix(x_train[xFeat]), label=y_train)
dval <- xgb.DMatrix(data.matrix(x_val[xFeat]), label=y_val)

#perform training
gbdt = xgb.train(params = xgb_params,
                 data = dtrain,
                 nrounds =500,
                 watchlist = list(train = dtrain, val=dval),
                 print_every_n = 25,
                 early_stopping_rounds=50)


train.testM <- xgb.DMatrix(data.matrix(train.test[xFeat]))

allpredictions =  (as.data.frame(matrix(predict(gbdt,train.testM), nrow=dim(train.testM), byrow=TRUE)))

##
OOF_prediction <- allpredictions[1:3] %>%
  mutate(max_prob = max.col(., ties.method = "last")) 
#View(OOF_prediction)
##
######################
##Generate Submission
allpredictions = cbind (allpredictions, train.test$listing_id)
names(allpredictions)<-c("high","low","medium","listing_id")
allpredictions=allpredictions[,c(1,3,2,4)]
write.csv(allpredictions,paste0(Sys.Date(),"-BaseModel-20Fold-Seed",seed,".csv"),row.names = FALSE)


####################################
###Generate Feature Importance Plot
imp <- xgb.importance(names(train.train[xFeat]),model = gbdt)
xgb.ggplot.importance(imp)

"class"

compare = as.data.frame(cbind(OOF_prediction$max_prob,train.test$interest_level))
colnames(compare) = c('predicted', 'real')
conf_matrix <- table(compare$predicted, compare$real)
confusionMatrix(conf_matrix)
