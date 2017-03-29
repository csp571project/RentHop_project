install.packages(c("purrr", "jsonlite", "dplyr"))
# setwd("/Users/shuyuzhang/Desktop/2017Spring/CSP 571/FinalProject/Data")

library(jsonlite)
library(purrr)
library(data.table)
library('lubridate')

train <- fromJSON("train.json")
test <- fromJSON("test.json")


sapply(train, class)
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)

str(train)
table(is.na(train))
summary(train)
# Created year is useless because they are all 2016
train$created_year <- format(as.Date(as.Date(train$created), format="%Y-%m-%d"),"%Y")

# created month also only has three value "06" "05" "04" (not quite useful)
train$created_month <- format(as.Date(as.Date(train$created), format="%Y-%m-%d"),"%m")

days <- as.Date(train$created)
train$weekend <- (wday(days,label = TRUE) == "Sat") | (wday(days,label = TRUE) == "Sun")

#get the value of interest_level
unique(train$interest_level)
#reorder the levels
train$interest_level <- factor(train$interest_level, levels =c("low","medium","high"))
# "low" "medium" "high" to 1 2 3 as numeric
train$interest_level_num <- as.numeric(train$interest_level)


table(train$interest_level)

write.csv(train[c('listing_id','bedrooms',
                  'bathrooms','price','created_month',
                  'weekend', 'interest_level', 'interest_level_num')], file = 'train_baseline.csv', row.names = FALSE)

