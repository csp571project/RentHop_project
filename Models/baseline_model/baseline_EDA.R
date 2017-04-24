library(nnet)
library(ggplot2)
library(lattice)

# train_base <- read.csv('../processed_data/train_baseline11_v2.csv',header = TRUE)
# train_base_add <- read.csv('../processed_data/train_baselineNEW.csv',header = TRUE)
train_base <- read.csv('../../processed_data/train_baselineCLEAN.csv',header = TRUE)
# train_base <- merge(train_base, train_base_add)
str(train_base)

# reorder interest levels
train_base$interest_level <- factor(train_base$interest_level, 
                                    levels = c("low", "medium", "high"))

# plot the distribution of interest level
ggplot(aes (x = interest_level), data = train_base) + 
  geom_bar(color = 'black', fill = "darkgreen", alpha = 0.5)
# ggsave("../Plots/Interest Level Distribution.png",width = 21, height = 20, units = 'cm')

# univariate 
# bedrooms
summary(train_base$bedrooms)
with(train_base, table(interest_level, bedrooms))

by(train_base$bedrooms, train_base$interest_level, summary)
ggplot(aes(x = bedrooms), data = train_base, binwidth = 1) + 
  geom_bar(color = 'black', aes(fill= interest_level), alpha = .5) +
  labs(title="Bar for Bedrooms") + scale_x_continuous(breaks=seq(0, 8, 1))
# ggsave("../Plots/Barplot of Bedrooms.png", width = 21, height = 20, units = 'cm')

# bathrooms
# outlier detection
train_base[which(train_base$bathrooms>2*train_base$bedrooms & train_base$bedrooms!=0), 2:3]

#        bedrooms bathrooms
# 2076         1       3.0
# 3088         1       3.0
# 4422         2      10.0
# 5385         1       3.0
# 13960        1       3.0
# 15422        1       4.5
# 24365        1       3.0
# 24695        1       3.0
# 24737        1       3.0
# 28838        1       3.0
# 33076        1       3.0
# 39257        1       3.0
# 41658        1       3.0
# 41744        1       2.5
# 45907        1       3.0
# 46689        1       2.5
# 47733        1       3.0
# 47935        1       3.0

by(train_base$bathrooms, train_base$interest_level, summary)
with(train_base, table(interest_level, bathrooms))

summary(train_base$bathrooms)

ggplot(aes(x = bathrooms), data = train_base) + 
  geom_histogram(color = 'black', aes(fill= interest_level), alpha = .5 )+
  labs(title="Barplot for Bathrooms") + coord_cartesian(xlim = c(0,6)) +
  scale_x_continuous(breaks=seq(0, 6, 0.5))
# ggsave("Barplot of Bathrooms.png", width = 21, height = 20, units = 'cm')

# price
by(train_base$price, train_base$interest_level, summary)
summary(train_base$price)
ggplot(train_base, aes(x=(price))) + 
  geom_histogram(color = 'black', aes(fill=interest_level), alpha = .5) + 
  xlim(c(1000,11000)) + labs(title = "Histogram of Bathrooms")
# ggsave("Histogram of Bathrooms.png", width = 21, height = 20, units = 'cm')


# number of photos
by(train_base$numPh, train_base$interest_level, summary)
summary(train_base$numPh)
ggplot(train_base, aes(x=(numPh)), binwidth = 1) + 
  geom_histogram(color = 'black', aes(fill=interest_level), alpha = .5) + xlim(c(0, 30)) +
  labs(title = "Histogram of Number of Photos")
# ggsave("Histogram of NumPhoto.png", width = 21, height = 20, units = 'cm')

# number of features
by(train_base$numFeat, train_base$interest_level, summary)
summary(train_base$numFeat)
ggplot(train_base, aes(x=(numFeat))) + 
  geom_histogram(color = 'black', aes(fill=interest_level), alpha = .5) + xlim(c(0,30)) +
  labs(title = "Histogram of Number of Features")

# ggsave("Histogram of NumFeatures.png", width = 21, height = 20, units = 'cm')


# distance to city
by(train_base$distance_city, train_base$interest_level, summary)

summary(train_base$distance_city)
ggplot(train_base, aes(x=(distance_city), colour = interest_level) ) + 
  geom_density() + xlim(c(0,0.3)) +
  labs(title = "Density Plot of Distance to City")
# ggsave("Density Plot of Distance to City.png", width = 21, height = 20, units = 'cm')


# created month
by(train_base$created_month, train_base$interest_level, summary)
qplot(created_month, data = train_base) + facet_wrap( ~ interest_level, scales = "free_y")

# created at weekend
by(train_base$weekend, train_base$interest_level, summary)
qplot(weekend, data = train_base) + facet_wrap( ~ interest_level, scales = "free_y")

# created hour
ggplot(aes(x = created_hour), data = train_base) + geom_bar(aes(fill = interest_level)) +
  scale_x_continuous(breaks=seq(0, 23, 1)) + labs(title = "Barplot of Created Hour")
ggsave("/Barplot of Created Hour.png", width = 40, height = 20, units = 'cm')

# time of day created
ggplot(aes(x = time_ofday), data = train_base) + geom_bar(aes(fill = interest_level)) +
  labs(title = "Barplot of Time of Day")
ggsave("Barplot of Time of Day.png", width = 40, height = 20, units = 'cm')

# Result shows that created_hour might effect the interest

# price and number of bedrooms
library(dplyr)
bedroom_group <- group_by(train_base, bedrooms)

train_base.price_by_bd <- train_base %>%
  group_by(bedrooms) %>%
  summarise(price_mean = mean(price),
            price_median = median(price), 
                            #number of users
            n = n())  %>%
  arrange(bedrooms)
head(train_base.price_by_bd)

# price and number of bathroom
bathroom_group <- group_by(train_base, bathrooms)

train_base.price_by_bath <- train_base %>%
  group_by(bathrooms) %>%
  summarise(price_mean = mean(price),
            price_median = median(price), 
            #number of users
            n = n())  %>%
  arrange(bathrooms)
head(train_base.price_by_bath)

library(corrplot)
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

# ------finished EDA----------

