library(dplyr)
library(ggplot2)
library(magrittr)
library(jsonlite)
library(ggmap)
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

ny_outliners_dist <- 0.2

ggplot(training[training$distance_city < ny_outliners_dist, ],
       aes(distance_city, color = interest_level)) +
  geom_density()


map <- get_googlemap(
  zoom = 12,
  # Use Alternate New York City Center Coords
  center = ny_center %>% as.numeric,
  maptype = "satellite",
  sensor = FALSE)

ggmap(map) +
  geom_point(size = 1,
             data = training,
             aes(x = longitude,
                 y = latitude,
                 color = interest_level)) +  scale_colour_brewer(palette = "Set1")

##Find if there are any outliers 
any(training$latitude == 0)
###Which are they
outliers_addrs <- training[training$longitude == 0 | training$latitude == 0, ]$street_address
outliers_addrs
outliers_ny <- paste(outliers_addrs, ", new york")
outliers_addrs <- data.frame("street_address" = outliers_addrs)
coords <- sapply(outliers_ny,
                 function(x) geocode(x, source = "google")) %>%
  t %>%
  data.frame %>%
  cbind(outliers_addrs, .)

rownames(coords) <- 1:nrow(coords)
# Display table
kable(coords)
