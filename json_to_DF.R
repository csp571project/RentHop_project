library(dplyr)
library(ggplot2)
library(magrittr)
library(jsonlite)
library(ggmap)
library(knitr)

################################################################################
##        READ DATA
################################################################################

training <- fromJSON("/home/raz/Downloads/train.json") %>% 
  bind_rows 
features <- training$features
photos <- training$photos

training$features <- NULL
training$photos <- NULL 

# Convert to data.frame
training <- sapply(training, unlist) %>%
  data.frame(., stringsAsFactors = FALSE)

# Add removed variables
training$features <- features
training$photos <- photos

training$numFeat <- NULL
training$numFeat <- NULL 

for(i in 1:length(training$photos))
{
  training$numPh[i] <- length(training$photos[[i]])
  training$numFeat[i] <- length(training$features[[i]])
}

numerical_variables <- c("bathrooms", "bedrooms",
                         "longitude", "latitude", "price")

training[, numerical_variables] %<>%
  lapply(., as.numeric)

training$interest_level <- as.factor(training$interest_level)


