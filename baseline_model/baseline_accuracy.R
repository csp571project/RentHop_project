
library(nnet)
library(caret)

train <- read.csv('train_baseline.csv',header = TRUE)


train_base=train[c('listing_id','bedrooms',
                   'bathrooms','price','created_month',
                   'weekend', 'interest_level', 'interest_level_num')]
inTrain <- createDataPartition(y = train_base$interest_level_num, list = FALSE, p = 0.8)

train_base.train <- train_base[inTrain,]
train_base.test <- train_base[-inTrain,]
t <- multinom(interest_level ~ bedrooms + bathrooms + price + weekend + created_month, data = train_base.train)
predicted = predict(t, train_base.test, type = "probs")
compare = cbind(apply(predicted,1, which.max),train_base.test$interest_level_num)


accuracy = sum(compare[,1]==compare[,2])/dim(compare)[1]
accuracy
