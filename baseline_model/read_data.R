# install.packages(c("purrr", "jsonlite", "dplyr"))
# setwd("/Users/shuyuzhang/Desktop/2017Spring/CSP 571/FinalProject/Data")

library(jsonlite)
library(purrr)
library(data.table)
library('lubridate')

train <- fromJSON("../train.json")
test <- fromJSON("../test.json")


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

# created day to logical value:  weekend or not 
days <- as.Date(train$created)
train$weekend <- (wday(days,label = TRUE) == "Sat") | (wday(days,label = TRUE) == "Sun")

# Adding created hour see if it can improve accuracy of baseline model
train$created_hour <- substr(train$created,12,13)

#get the value of interest_level
unique(train$interest_level)

#reorder the levels
train$interest_level <- factor(train$interest_level, levels =c("low","medium","high"))
# "low" "medium" "high" to 1 2 3 as numeric
train$interest_level_num <- as.numeric(train$interest_level)

# adding manger_score according to 
manager_id <- unique(train$manager_id)
manager_score <- NULL
for(i in 1:length(manager_id)){
  id <- manager_id[i]
  total <- train$interest_level_num[train$manager_id == id]
  manager_score[i] <- sum(total) / length(total)
}

manager_score <- cbind(manager_id, manager_score)
train <- merge(train, manager_score)


table(train$interest_level)

write.csv(train[c('listing_id','bedrooms',
                  'bathrooms','price','created_month', 'created_hour',
                  'weekend', 'interest_level', 'interest_level_num', 'manager_score')], file = '../processed_data/train_baselineNEW.csv', row.names = FALSE)

