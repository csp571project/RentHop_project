# install.packages(c("purrr", "jsonlite", "dplyr"))
# setwd("/Users/shuyuzhang/Desktop/2017Spring/CSP 571/FinalProject/Data")

library(jsonlite)
library(purrr)
library(data.table)
library('lubridate')

train <- fromJSON('../train.json')

sapply(train, class)
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)

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

# adding manger_score using the sum of interest_level_num devided by the number of building under
# a specific manager_id
manager_id <- unique(train$manager_id)
manager_score <- NULL
for(i in 1:length(manager_id)){
  id <- manager_id[i]
  total <- train$interest_level_num[train$manager_id == id]
  manager_score[i] <- sum(total) / length(total)
}
manager_mean = mean(manager_score)
manager_score <- cbind(manager_id, as.numeric(manager_score))
train <- merge(train, manager_score)

# building_score using the same method as manager_score
building_id <- unique(train$building_id)
building_score <- NULL
for(i in 1:length(building_id)){
  id <- building_id[i]
  total <- train$interest_level_num[train$building_id == id]
  building_score[i] <- sum(total) / length(total)
}
building_mean = mean(building_score)
building_score <- cbind(building_id, building_score)
train <- merge(train, building_score)


table(train$interest_level)

write.csv(train[c('listing_id','bedrooms',
                  'bathrooms','price','created_month', 'created_hour',
                  'weekend', 'interest_level', 'interest_level_num', 
                  'manager_score', 'building_score')],
          file = '../processed_data/train_baselineNEW.csv', row.names = FALSE)

##############################################################
test <- fromJSON("../test.json")
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
                                                         
test$created_year <- format(as.Date(as.Date(test$created), format="%Y-%m-%d"),"%Y")
test$created_month <- format(as.Date(as.Date(test$created), format="%Y-%m-%d"),"%m")

days <- as.Date(test$created)
test$weekend <- (wday(days,label = TRUE) == "Sat") | (wday(days,label = TRUE) == "Sun")

test$created_hour <- substr(test$created,12,13)

test = merge(test, manager_score, by='manager_id', all.x = TRUE)
test$manager_score = as.numeric(test$V2)
test$V2 = NULL
test$manager_score[is.na(test$manager_score)] = manager_mean

test = merge(test, building_score, by = 'building_id', all.x = TRUE)
test$building_score = as.numeric(test$building_score)
#test$V2 = NULL
test$building_score[is.na(test$building_score)] = building_mean

# Add  the number of features and photos
for(i in 1:length(test$photos)){
  test$numPh[i] <- length(test$photos[[i]])
  test$numFeat[i] <- length(test$features[[i]])
}

# Add distance to center
#install.packages('ggmap')
library(ggmap)

# New York City Center Coords
ny_center <- geocode("new york", source = "google")
ny_lat <-  ny_center[2]
ny_lon <-  ny_center[1]

# Add Euclidean Distance to City Center
test$distance_city <-
  mapply(function(lon, lat) sqrt((lon - ny_lon)^2  + (lat - ny_lat)^2),
         test$longitude, test$latitude) 

write.csv(test[c('listing_id','bedrooms',
                  'bathrooms','price','created_month', 'created_hour',
                  'weekend', 'numPh', 'numFeat', 'distance_city',
                  'manager_score', 'building_score')],
          file = '../processed_data/test_baseline11_v1.csv', row.names = FALSE)
