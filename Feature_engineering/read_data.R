# install.packages(c("purrr", "jsonlite", "dplyr"))
# setwd("/Users/shuyuzhang/Desktop/2017Spring/CSP 571/FinalProject/Data")

library(jsonlite)
library(purrr)
library(data.table)
library('lubridate')

train <- fromJSON('.//train.json')


sapply(train, class)
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)

names(train)
table(is.na(train))
summary(train)
###############
####################
transfer=list('low'=1, 'medium'=2, 'high'=3)

# Construct the basic description data frame
description=data.frame(cbind(train$listing_id,
                             train$description,
                             train$interest_level),
                       stringsAsFactors=FALSE)
colnames(description) = c('listing_id', 'origin_des', 'interest_level')

description$interest_Nbr <- unlist(sapply(description$interest_level,
                                          function(v) transfer[v]), use.names = FALSE)

####################
## plain description is to remove the suffix
# Get rid of the html patterns inside the text
# https://tonybreyal.wordpress.com/2011/11/18/htmltotext-extracting-text-from-html-via-xpath/
pattern = "</?\\w+((\\s+\\w+(\\s*=\\s*(?:\".*?\"|'.*?'|[^'\">\\s]+))?)+\\s*|\\s*)/?>"
description$plain_des = gsub(pattern, "\\1", description$origin_des)

# Get rid of the ending word
description$plain_des = gsub("website_redacted", " ", description$plain_des)
description$plain_des = gsub("<a", " ", description$plain_des)

# This wordcount function does not distinguish duplicated words and will count the blank description as 0 word
wordcount = function(v){
  temp = strsplit(v, "\\W+")[[1]]
  if(sum(which(temp==""))>0)
    temp = temp[-which(temp=="")]
  return(length(temp))
}
description$plain_wordcount = vapply(description$plain_des, wordcount, integer(1), USE.NAMES = FALSE)

####################
## clean description is to remove the stopwords and punctuation
#install.packages("tm", dependencies = TRUE)
library('tm')
temp = VCorpus(VectorSource(description$plain_des))

# Transfer to lower case
corpus = tm_map(temp, content_transformer(tolower))
# Get rid of the stop words
corpus = tm_map(corpus, removeWords, stopwords("english"))
(f <- content_transformer(function(x, pattern) gsub(pattern, " ", x)))
# Get rid of the signle alphabetic and digit
corpus = tm_map(corpus, f, "[[:digit:]]")
corpus = tm_map(corpus, f, " *\\b[[:alpha:]]{1}\\b *")
# Get rid of punctuations
corpus = tm_map(corpus, f, "[[:punct:]]")
# Get rid of extra white spaces
corpus = tm_map(corpus, stripWhitespace)

description$clean_des = unlist(lapply(corpus, as.character))

train$clean_wordcount = vapply(description$clean_des, wordcount, integer(1), USE.NAMES=FALSE)

remove(description)
remove(corpus)
gc()

###############
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


table(train$interest_level)

##RP

train$numPh <- 0
train$numFeat <- 0
# Add  the number of features and photos
for(i in 1:length(train$photos)){
  train$numPh[i] <- length(train$photos[[i]])
  train$numFeat[i] <- length(train$features[[i]])
}

# Add distance to center
#install.packages('ggmap')
library(ggmap)

# New York City Center Coords
ny_center <- geocode("new york", source = "google")
ny_lat <-  ny_center[2]
ny_lon <-  ny_center[1]

train$distance_city <- 0
# Add Euclidean Distance to City Center
train$distance_city <-
  mapply(function(lon, lat) sqrt((lon - ny_lon)^2  + (lat - ny_lat)^2),
         train$longitude, train$latitude) 

names(train)

time.tag <-  c("00", "06", "12", "18", "24")

train$time_ofday <- cut(as.numeric(train$created_hour), breaks=time.tag,
                        labels = c("00-06","06-12", "12-18", "18-23"), include.lowest= TRUE)

train$time_ofday <- as.factor(train$time_ofday)


##################################################################################################################

#Removing duplicate values from features (combining related words)


train$features<-sapply(train$features, tolower)


#Group related features
train$features <- lapply(train$features, function(x) {gsub("hardwood floor|^wood|^floors|^floor|hardwoodfloor", "hardwood", x)})

train$features <- lapply(train$features, function(x) {gsub("dryer\\.|washer\\.|dryer in unit|washer/dryer|washer in unit|laundry in building|laundry in unit", "laundry in unit", x)})

train$features <- lapply(train$features, function(x) {gsub("roof\\.|common rooftop|roof-deck|common roof deck", "rooftop", x)})

train$features <- lapply(train$features, function(x) {gsub("^outdoor\\.", "outdoor space", x)})

train$features <- lapply(train$features, function(x) {gsub("^garden\\.", "garden", x)})

train$features <- lapply(train$features, function(x) {gsub("^pre war|pre-war", "war", x)})
train$features <- lapply(train$features, function(x) {gsub("^pool|^swimmingpool|^swimming pool", "swimming", x)})

train$features <- lapply(train$features, function(x) {gsub("^park\\.|common parking/garage", "parking", x)})

#Keep only unique features
train$features<-lapply(train$features, function(x) {unique(unlist(x, use.names = FALSE))})

head(train$features,10)

word_remove = c('allowed', 'building','center', 'space','2','2br','bldg','24',
                '3br','1','ft','3','7','1br','hour','bedrooms','bedroom','true',
                'stop','size','blk','4br','4','sq','0862','1.5','373','16','3rd','block',
                'st','01','bathrooms','unit','room','pool')

feat <- c("bathrooms","bedrooms","building_id", "created","latitude", "description",
          "listing_id","longitude","manager_id", "price", "features",
          "display_address", "street_address","numFeat","numPh", "interest_level")


train1 = train[,names(train) %in% feat]

#create sparse matrix for word features
word_sparse<-train1[,names(train1) %in% c("features","listing_id")]
#train1["features_seg"] = NULL


#Create word features

library(tidytext)
word_sparse <- word_sparse %>%
  filter(map(features, is_empty) != TRUE) %>%
  tidyr::unnest(features) %>%
  unnest_tokens(word, features)

data("stop_words")

#remove stop words and other words
word_sparse = word_sparse[!(word_sparse$word %in% stop_words$word),]
word_sparse = word_sparse[!(word_sparse$word %in% word_remove),]

#get most common features and use (in this case top 25)
top_word <- as.character(as.data.frame(sort(table(word_sparse$word),decreasing = TRUE)[1:10])$Var1)
word_sparse = word_sparse[word_sparse$word %in% top_word,]
word_sparse$word = as.factor(word_sparse$word)
word_sparse<-dcast(word_sparse, listing_id ~ word,length, value.var = "word")

#merge word features back into main data frame
train<-merge(train,word_sparse, by = "listing_id", sort = FALSE,all.x=TRUE)


apply(train,2,anyNA)

train[is.na(train)] <- 0

head(train[,25:35])


write.csv(train[c('listing_id','bedrooms',
                  'bathrooms','price','created_month', 'created_hour',
                  'weekend', 'interest_level', 'interest_level_num', 'time_ofday' , 'clean_wordcount',
                  'numPh', 'numFeat', 'distance_city',"cats","dishwasher","dogs","doorman","elevator","fee","fitness"
                  ,"hardwoods","laundry","war")],
          file = '../../processed_data/train_baselineCLEAN.csv', row.names = FALSE)






##############################################################

#Train-test split


set.seed(11080)
library('caret')
trainsplit <- 0.8

testsplit <- 1 - trainsplit
inTrain <- createDataPartition(y = train$interest_level, p = trainsplit, list = FALSE)
renttrain <- train[inTrain,]
renttest <- train[-inTrain,]
stopifnot(nrow(renttrain) + nrow(renttest) == nrow(train))

remove(ny_center,ny_lat, ny_lon, inTrain, stop_words, word_sparse,
       days, f,feat, i, pattern, temp, testsplit, transfer, time.tag, top_word, trainsplit,
       vars, word_remove)
remove(train1)

gc()




















# test <- fromJSON("../test.json")
# test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
#                                                          
# test$created_year <- format(as.Date(as.Date(test$created), format="%Y-%m-%d"),"%Y")
# test$created_month <- format(as.Date(as.Date(test$created), format="%Y-%m-%d"),"%m")
# 
# days <- as.Date(test$created)
# test$weekend <- (wday(days,label = TRUE) == "Sat") | (wday(days,label = TRUE) == "Sun")
# 
# test$created_hour <- substr(test$created,12,13)
# 
# test = merge(test, manager_score, by='manager_id', all.x = TRUE)
# test$manager_score = as.numeric(test$V2)
# test$V2 = NULL
# test$manager_score[is.na(test$manager_score)] = manager_mean
# 
# # Check the unique manager and building amount in train and test
# length(names(table(train$manager_id)))
# #[1] 3481
# length(names(table(test$manager_id)))
# #[1] 3851
# 
# length(names(table(train$building_id)))
# #[1] 7585
# length(names(table(test$building_id)))
# #[1] 9321
# 
# length(setdiff(names(table(test$manager_id)), names(table(train$manager_id))))
# #[1] 918
# length(setdiff(names(table(train$manager_id)), names(table(test$manager_id))))
# #[1] 548
# length(setdiff(names(table(test$manager_id)), names(table(train$manager_id))))/length(names(table(test$manager_id)))
# #[1] 0.2383796
# 
# length(setdiff(names(table(test$building_id)), names(table(train$building_id))))
# #[1] 4050
# length(setdiff(names(table(train$building_id)), names(table(test$building_id))))
# #[1] 2314
# length(setdiff(names(table(test$building_id)), names(table(train$building_id))))/length(names(table(test$building_id)))
# #[1] 0.4345027
# 
# 
# test = merge(test, building_score, by = 'building_id', all.x = TRUE)
# test$building_score = as.numeric(test$building_score)
# #test$V2 = NULL
# test$building_score[is.na(test$building_score)] = building_mean
# 
# # Add  the number of features and photos
# for(i in 1:length(test$photos)){
#   test$numPh[i] <- length(test$photos[[i]])
#   test$numFeat[i] <- length(test$features[[i]])
# }
# 
# # Add distance to center
# #install.packages('ggmap')
# library(ggmap)
# 
# # New York City Center Coords
# ny_center <- geocode("new york", source = "google")
# ny_lat <-  ny_center[2]
# ny_lon <-  ny_center[1]
# 
# # Add Euclidean Distance to City Center
# test$distance_city <-
#   mapply(function(lon, lat) sqrt((lon - ny_lon)^2  + (lat - ny_lat)^2),
#          test$longitude, test$latitude) 
# 
# write.csv(test[c('listing_id','bedrooms',
#                   'bathrooms','price','created_month', 'created_hour',
#                   'weekend', 'numPh', 'numFeat', 'distance_city',
#                   'manager_score', 'building_score')],
#           file = '../processed_data/test_baseline11_v1.csv', row.names = FALSE)
