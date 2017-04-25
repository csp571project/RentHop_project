library(nnet)
library(caret)


########################
# load all processed data
########################
setwd("/home/raz/Projects/TEMP/EDA_features/Models")
# train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
# train_base1  <- read.csv('../processed_data/train_baselineNEW.csv', header = TRUE)
train <- read.csv('../processed_data/train_moreFeat36.csv', header = TRUE)

########################
createModelFormula <- function(targetVar, xVars, includeIntercept = TRUE){
  if(includeIntercept){
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ ')))
  } else {
    modelForm <- as.formula(paste(targetVar, "~", paste(xVars, collapse = '+ '), -1))
  }
  return(modelForm)
}


xFeat <- names(train)[c(-1, -9)]
yFeat <- "interest_level"

modelForm <- createModelFormula(yFeat, xFeat[-31])



require(doMC)
require(caret)
registerDoMC(cores=6)
library(randomForest)


train$distance_city<- log(train$distance_city) 
# modelForm1 <- createModelFormula(targetVar, c(catVars, numVarsfit))

control <- trainControl(method="adaptive_cv", number=8, repeats=5, search="grid", verboseIter = TRUE,allowParallel=TRUE , returnResamp="all")
fit1 <-  randomForest(modelForm, data = train, importance = TRUE, 
                      ntree = 1000, allowParallel = TRUE, trcontrol = rf_ctrl, verbose=TRUE)


gc()
ktest<- read.csv("../processed_data/test_moreFeat36.csv", header = TRUE)
ktest$distance_city <- log(ktest$distance_city) 
allpredictions      <- cbind (predsFit1, ktest$listing_id)
colnames(allpredictions)<-c("high","low","medium","listing_id")
allpredictions=allpredictions[,c(1,3,2,4)]
write.csv(allpredictions,paste0(Sys.Date(),"-rpawar2RF_FINXAAAA",seed,".csv"),row.names = FALSE)

var_importance <- data_frame(variable=setdiff(colnames(train[-1]), "interest_level"),
                             importance=as.vector(importance(fit1)[,5]))
var_importance <- arrange(var_importance, desc(importance))
var_importance$variable <- factor(var_importance$variable, levels=var_importance$variable)

p <- ggplot(var_importance, aes(x=variable, weight=importance, fill=variable))
p <- p + geom_bar() + ggtitle("Variable Importance from Random Forest Fit")
p <- p + xlab("Demographic Attribute") + ylab("Variable Importance (Mean Decrease in Gini Index)")
p <- p + scale_fill_discrete(name="Variable Name")
p + theme(axis.text.x=element_blank(),
          axis.text.y=element_text(size=12),
          axis.title=element_text(size=16),
          plot.title=element_text(size=18),
          legend.title=element_text(size=16),
          legend.text=element_text(size=12))

