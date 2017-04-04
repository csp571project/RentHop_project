library(nnet)

train_base <- read.csv('../processed_data/train_baseline11_v2.csv',header = TRUE)

# plot the distribution of bedrooms and bathrooms
hist(train$bathrooms)
hist(train$bedrooms)

# print the coefficients
cor(train$interest_level_num, train$bedrooms)
cor(train$interest_level_num, train$bathrooms)
cor(train$interest_level_num, train$weekend)
table(train$interest_level_num, as.factor(train$weekend))

#   FALSE  TRUE
# 1 26080  8204
# 2  8800  2429
# 3  2978   861
# FALSE is weekday 
# TRUE is for weekend

with(train, table(interest_level, bedrooms))
with(train, table(interest_level, bathrooms))


histogram(~bathrooms+bedrooms | interest_level, train)

# build baseline model
t <- multinom(interest_level ~ bedrooms + bathrooms + price + weekend + created_month, data = train)

# ------finished EDA----------

