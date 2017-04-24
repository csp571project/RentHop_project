#Run decision trees on the selected features
#renttrain<-renttrain["interest_level_num"]
renttrain<-subset(renttrain, select=-c(interest_level_num))
y<-renttest$interest_level
renttest<-subset(renttest, select=-c(interest_level_num,interest_level))

library(rpart)
fit3 <- rpart(interest_level~.,
              data=renttrain,
              method="class",
              control=rpart.control(minsplit=150, cp=0.001))
printcp(fit3)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

fancyRpartPlot(fit3)


Prediction <- predict(fit3, renttest, type = "class")
Actual <- y

confusionMatrix(reference = Actual, data = Prediction)

#Highest accuracy with different combinations of rpart.control parameters
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  low medium high
# low    6527   1736  433
# medium  304    447  216
# high     25     62  118
# 
# Overall Statistics
# 
# Accuracy : 0.7187          
# 95% CI : (0.7097, 0.7275)
# No Information Rate : 0.6948          
# P-Value [Acc > NIR] : 1.074e-07       
# 
# Kappa : 0.2268          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: low Class: medium Class: high
# Sensitivity              0.9520       0.19911     0.15385
# Specificity              0.2799       0.93179     0.99044
# Pos Pred Value           0.7506       0.46225     0.57561
# Neg Pred Value           0.7193       0.79800     0.93284
# Prevalence               0.6948       0.22750     0.07773
# Detection Rate           0.6614       0.04530     0.01196
# Detection Prevalence     0.8812       0.09799     0.02077
# Balanced Accuracy        0.6159       0.56545     0.57214