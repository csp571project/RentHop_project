library(nnet)
library(caret)


########################
# load all processed data
########################
setwd("/home/raz/Projects/TEMP/EDA_features/Models")
# train_base <- read.csv('../processed_data/train_baseline11_v2.csv', header = TRUE)
# train_base1  <- read.csv('../processed_data/train_baselineNEW.csv', header = TRUE)
train <- read.csv('../processed_data/train_moreFeat36.csv', header = TRUE)
#sentim <-  read.csv('../processed_data/train_description_sentiment.csv', header = TRUE)
#View(sentim)
#train <-merge(train,sentim , by = "listing_id")
View(train)

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

featExtract <- function(c, lvl) {
  ###lvl controls what number of count to be considered as factors for numerics
  if (missing(lvl)) {
    lvl <- 4
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

catFeat<- c(catFeat, "fee", "fitness", "laundry")

train[numFeat] <- lapply(train[numFeat], as.numeric)
train[catFeat] <- lapply(train[catFeat], as.factor)
train[binFeat] <- lapply(train[binFeat], as.factor)

train <- droplevels(train)

View(train)
train$interest_level<-as.integer(factor(train$interest_level))

control <- trainControl(method="adaptive_cv",
                        number=8,
                        repeats=5,
                        search="grid",
                        verboseIter = TRUE,allowParallel=TRUE , returnResamp="all")
fit1 <-  randomForest(modelForm, data = train, importance = TRUE, 
                      ntree = 600, allowParallel = TRUE, trcontrol = control, verbose=TRUE)


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

tsne = Rtsne(as.matrix(train), check_duplicates=FALSE, pca=TRUE, 
             perplexity=30, theta=0.5, dims=2)

