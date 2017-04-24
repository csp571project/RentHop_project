library(nnet)
library(ggplot2)
library(lattice)

train_base <- read.csv('../../processed_data/train_baseline11_v2.csv',header = TRUE)
str(train_base)

# reorder interest levels
train_base$interest_level <- factor(train_base$interest_level, 
                                    levels = c("low", "medium", "high"))
# univariate 
# bedrooms
summary(train_base$bedrooms)
with(train_base, table(interest_level, bedrooms))

by(train_base$bedrooms, train_base$interest_level, summary)
ggplot(aes(x = bedrooms), data = train_base, binwidth = 1) + 
  geom_histogram(color = 'black', aes(fill=..count..), 
                 breaks=seq(0, 8, by = 1), alpha = .2) +
  labs(title="Histogram for Bedrooms") +
  labs(x="Bedrooms", y="Count") +  scale_fill_gradient("Count", low = "green", high = "red") + 
  facet_wrap( ~ interest_level)

# bathrooms
by(train_base$bathrooms, train_base$interest_level, summary)
with(train_base, table(interest_level, bathrooms))

summary(train_base$bathrooms)
ggplot(aes(x = bathrooms), data = train_base, binwidth = 1) + 
  geom_histogram(color = 'black', aes(fill=..count..), 
                 breaks=seq(0, 8, by = 1), alpha = .2) +
  labs(title="Histogram for Bathrooms") +
  labs(x="Bathrooms", y="Count") +  scale_fill_gradient("Count", low = "green", high = "red") + 
  facet_wrap( ~ interest_level)

# price
by(train_base$price, train_base$interest_level, summary)
summary(train_base$price)
ggplot(train_base, aes(x=(price))) + 
  geom_histogram(color = 'black', aes(fill=..count..), alpha = .2) + 
  xlim(c(1000,11000)) + facet_wrap( ~ interest_level)

# number of photos
by(train_base$numPh, train_base$interest_level, summary)
summary(train_base$numPh)
ggplot(train_base, aes(x=(numPh)), binwidth = 1) + 
  geom_histogram(color = 'black', aes(fill=..count..), alpha = .2) + xlim(c(0, 30)) + 
  facet_wrap( ~ interest_level)

# number of features
by(train_base$numFeat, train_base$interest_level, summary)
summary(train_base$numFeat)
ggplot(train_base, aes(x=(numFeat))) + 
  geom_histogram(color = 'black', aes(fill=..count..), alpha = .2) + xlim(c(0,30)) +
  facet_wrap( ~ interest_level)


# distance to city
by(train_base$distance_city, train_base$interest_level, summary)
summary(train_base$distance_city)
ggplot(train_base, aes(x=(distance_city))) + 
  geom_histogram(color = 'black', aes(fill=..count..), alpha = .2) + xlim(c(0,1)) + 
  facet_wrap( ~ interest_level)

# created month
by(train_base$created_month, train_base$interest_level, summary)
qplot(created_month, data = train_base) + facet_wrap( ~ interest_level)

# created at weekend
by(train_base$weekend, train_base$interest_level, summary)
qplot(weekend, data = train_base) + facet_wrap( ~ interest_level)

# There is no effective evidence showing that interest_level is related to the created_date


# print the coefficients
cor(train_base$interest_level_num, train_base$bedrooms)
cor(train_base$interest_level_num, train_base$bathrooms)
cor(train_base$interest_level_num, train_base$weekend)
table(train_base$interest_level_num, as.factor(train_base$weekend))

#   FALSE  TRUE
# 1 26080  8204
# 2  8800  2429
# 3  2978   861
# FALSE is weekday 
# TRUE is for weekend




# 
# histogram(~bathrooms+bedrooms | interest_level, train_base)

# build baseline model
t <- multinom(interest_level ~ bedrooms + bathrooms + price + weekend + created_month, data = train_base)

# ------finished EDA----------

