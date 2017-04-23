library(dplyr)
library(ggplot2)
library(magrittr)
library(jsonlite)
library(ggmap)

#  /home/raz/anaconda3/lib/R/library

library(knitr)

################################################################################
##        PROCESS_LOCATION_FEATURES
################################################################################

# Alternate New York City Center Coords

ny_center <- geocode("new york", source = "google")

# New York City Center Coords
ny_lat <-  geocode("new york", source = "google")[2]
ny_lon <-  geocode("new york", source = "google")[1]

# Add Euclidean Distance to City Center
training$distance_city <-
  mapply(function(lon, lat) sqrt((lon - ny_lon)^2  + (lat - ny_lat)^2),
         training$longitude,       training$latitude) 

ny_outliers_dist <- 0.2

###############################################################################

###############################################################################
data2 <- read.csv("/home/raz/Projects/EDA_features/baseline_model/train_baseline.csv", header = TRUE)

data3 <- merge(x = training[c("listing_id", "numPh", "numFeat", "distance_city")], y = data2, by = 'listing_id' )
View(data3)
################################################################################
### new baseline
################################################################################
dim(data3)
write.csv(data3, file = "/home/raz/Projects/EDA_features/processed_data/baseline11_v2.csv")




