library(jsonlite)
library(purrr)
library(data.table)
library('lubridate')


require(doMC)
registerDoMC(cores=4)

test <- fromJSON('~/Downloads/train.json')



vars <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
###############
####################
transfer = list('low' = 1,
                'medium' = 2,
                'high' = 3)

# Construct the basic description data frame
description = data.frame(
  cbind(test$listing_id,
        test$description),
  stringsAsFactors = FALSE
)
colnames(description) = c('listing_id', 'origin_des')

description$interest_Nbr <-
  unlist(sapply(description$interest_level,
                function(v)
                  transfer[v]),
         use.names = FALSE)

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
wordcount = function(v) {
  temp = strsplit(v, "\\W+")[[1]]
  if (sum(which(temp == "")) > 0)
    temp = temp[-which(temp == "")]
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
(f <-
    content_transformer(function(x, pattern)
      gsub(pattern, " ", x)))
# Get rid of the signle alphabetic and digit
corpus = tm_map(corpus, f, "[[:digit:]]")
corpus = tm_map(corpus, f, " *\\b[[:alpha:]]{1}\\b *")
# Get rid of punctuations
corpus = tm_map(corpus, f, "[[:punct:]]")
# Get rid of extra white spaces
corpus = tm_map(corpus, stripWhitespace)

description$clean_des = unlist(lapply(corpus, as.character))
test$clean_wordcount = vapply(description$clean_des, wordcount, integer(1), USE.NAMES =
                                 FALSE)

remove(description)
remove(corpus)
gc()

############### Shuyu 
test$created_year <-
  format(as.Date(as.Date(test$created), format = "%Y-%m-%d"), "%Y")

# created month also only has three value "06" "05" "04" (not quite useful)
test$created_month <-
  format(as.Date(as.Date(test$created), format = "%Y-%m-%d"), "%m")

# created day to logical value:  weekend or not
days <- as.Date(test$created)
test$weekend <-
  (wday(days, label = TRUE) == "Sat") |
  (wday(days, label = TRUE) == "Sun")

# Adding created hour see if it can improve accuracy of baseline model
test$created_hour <- substr(test$created, 12, 13)


##RP ######
# Add  the number of features and photos
test$numFeat <- lengths(test$features)
test$numPh <- lengths(test$photos)
# Add distance to center
#install.packages('ggmap')
library(ggmap)

# New York City Center Coords
ny_center <- geocode("new york", source = "google")
ny_lat <-  ny_center[2]
ny_lon <-  ny_center[1]

test$distance_city <- 0
# Add Euclidean Distance to City Center
test$distance_city <-
  mapply(function(lon, lat)
    sqrt((lon - ny_lon) ^ 2  + (lat - ny_lat) ^ 2),
    test$longitude, test$latitude)

time.tag <-  c("00", "06", "12", "18", "24")

test$time_ofday <-
  cut(
    as.numeric(test$created_hour),
    breaks = time.tag,
    labels = c("00-06", "06-12", "12-18", "18-23"),
    include.lowest = TRUE
  )

test$time_ofday <- as.factor(test$time_ofday)
##############################################################

#convert building and manager id to integer
test$building_id<-as.integer(factor(unlist(test$building_id)))
test$manager_id<-as.integer(factor(unlist(test$manager_id)))

#convert street and display address to integer
test$display_address<-as.integer(factor(unlist(test$display_address)))
test$street_address<-as.integer(factor(unlist(test$street_address)))


test$bed_price <- as.numeric(test$price)/as.numeric(test$bedrooms)
test[which(is.infinite(test$bed_price)),]$bed_price = test[which(is.infinite(test$bed_price)),]$price

#add sum of rooms and price per room
test$room_sum <- test$bedrooms + test$bathrooms
test$room_diff <- test$bedrooms - test$bathrooms
test$room_price <- test$price/test$room_sum
test$bed_ratio <- test$bedrooms/test$room_sum
test[which(is.infinite(test$room_price)),]$room_price = test[which(is.infinite(test$room_price)),]$price



#log transform features, these features aren't normally distributed
test$numPh <- log(test$numPh + 1)
test$numFeat <- log(test$numFeat + 1)
test$price <- log(test$price + 1)
test$room_price <- log(test$room_price + 1)
test$bed_price <- log(test$bed_price + 1)


##############################################################
## DD
#Removing duplicate values from features (combining related words)


test$features <- sapply(test$features, tolower)

#Group related features
test$features <-
  lapply(test$features, function(x) {
    gsub("hardwood floor|^wood|^floors|^floor|hardwoodfloor",
         "hardwood",
         x)
  })

test$features <-
  lapply(test$features, function(x) {
    gsub(
      "dryer\\.|washer\\.|dryer in unit|washer/dryer|washer in unit|laundry in building|laundry in unit",
      "laundry in unit",
      x
    )
  })

test$features <-
  lapply(test$features, function(x) {
    gsub("roof\\.|common rooftop|roof-deck|common roof deck",
         "rooftop",
         x)
  })

test$features <-
  lapply(test$features, function(x) {
    gsub("^outdoor\\.", "outdoor space", x)
  })

test$features <-
  lapply(test$features, function(x) {
    gsub("^garden\\.", "garden", x)
  })

test$features <-
  lapply(test$features, function(x) {
    gsub("^pre war|pre-war", "war", x)
  })
test$features <-
  lapply(test$features, function(x) {
    gsub("^pool|^swimmingpool|^swimming pool", "swimming", x)
  })

test$features <-
  lapply(test$features, function(x) {
    gsub("^park\\.|common parking/garage", "parking", x)
  })

#Keep only unique features
test$features <-
  lapply(test$features, function(x) {
    unique(unlist(x, use.names = FALSE))
  })

word_remove = c(
  'allowed',
  'building',
  'center',
  'space',
  '2',
  '2br',
  'bldg',
  '24',
  '3br',
  '1',
  'ft',
  '3',
  '7',
  '1br',
  'hour',
  'bedrooms',
  'bedroom',
  'true',
  'stop',
  'size',
  'blk',
  '4br',
  '4',
  'sq',
  '0862',
  '1.5',
  '373',
  '16',
  '3rd',
  'block',
  'st',
  '01',
  'bathrooms',
  'unit',
  'room',
  'pool'
)

feat <-
  c(
    "bathrooms",
    "bedrooms",
    "building_id",
    "created",
    "latitude",
    "description",
    "listing_id",
    "longitude",
    "manager_id",
    "price",
    "features",
    "display_address",
    "street_address",
    "numFeat",
    "numPh",
    "interest_level"
  )


test1 <- test[,names(test) %in% feat]

############################
##Process Word features
word_remove = c('allowed', 'building','center', 'space','2','2br','bldg','24',
                '3br','1','ft','3','7','1br','hour','bedrooms','bedroom','true',
                'stop','size','blk','4br','4','sq','0862','1.5','373','16','3rd','block',
                'st','01','bathrooms','unit','room','pool', 'pre')

#create sparse matrix for word features
word_sparse<-test1[,names(test1) %in% c("features","listing_id")]
test1$features = NULL
#word_sparse <- as.matrix(word_sparse)
require("tidytext")
require("tidyr")
#Create word features
word_sparse <- word_sparse                %>%
  tidyr::unnest(features) %>%
  tidytext::unnest_tokens(word, features)

data("stop_words")


#remove stop words and other words
word_sparse = word_sparse[!(word_sparse$word %in% stop_words$word), ]
word_sparse = word_sparse[!(word_sparse$word %in% word_remove), ]

#get most common features and use (in this case top 25)
top_word <-
  as.character(as.data.frame(sort(table(word_sparse$word), decreasing = TRUE)[1:10])$Var1)
word_sparse = word_sparse[word_sparse$word %in% top_word, ]
word_sparse$word = as.factor(word_sparse$word)
word_sparse <-
  dcast(word_sparse, listing_id ~ word, length, value.var = "word")

#merge word features back into main data frame
test <-
  merge(test,
        word_sparse,
        by = "listing_id",
        sort = FALSE,
        all.x = TRUE)


apply(test, 2, anyNA)

test[is.na(test)] <- 0

NonFeats <- c("created", "description", "features" , "latitude" , "longitude", "photos" , "created_year")

wrcsv <- test[ , !(names(test) %in% NonFeats)]

write.csv(wrcsv,file = '../processed_data/train_moreFeat36.csv', row.names = FALSE)



remove(
  ny_center,
  ny_lat,
  ny_lon,
  intest,
  stop_words,
  word_sparse,
  days,
  f,
  feat,
  i,
  pattern,
  temp,
  testsplit,
  transfer,
  time.tag,
  top_word,
  testsplit,
  vars,
  word_remove
)

remove(test, test1)
gc()
