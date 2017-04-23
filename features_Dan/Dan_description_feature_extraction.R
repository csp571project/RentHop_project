transfer=list('low'=1, 'medium'=2, 'high'=3)

####################
# # Read from train data set
library("rjson")
train_data<-fromJSON('../train.json')
train_data <- lapply(train_data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x,use.names = FALSE)
})
test_file<-files[1]
test_data<-fromJSON(file=test_file)
test_data <- lapply(test_data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x,use.names = FALSE)
})

####################
# Construct the basic description data frame
description=data.frame(cbind(train_data$listing_id,
                             train_data$description,
                             train_data$interest_level),
                       stringsAsFactors=FALSE)
colnames(description) = c('listing_id', 'origin_des', 'interest_level')
description$interest_Nbr <- unlist(sapply(description$interest_level,
                                          function(v) transfer[v]),
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
description$clean_wordcount = vapply(description$clean_des, wordcount, integer(1), USE.NAMES=FALSE)

write.csv(description, file= "train_description.csv", row.names=FALSE)

####################
## tokenize and tfidf
# Select the rows with the description length more than 0, tfidf only works on these rows
non_empty_clean_des = description[which(description$clean_wordcount!=0), ]

# Set the control of min wordlength to 1. To cover the description with only 1 word
# Here the single word description is not one signle character word, so the most frequent terms 
# won't cover word as 'a', '1', etc.
corpus = VCorpus(VectorSource(non_empty_clean_des$clean_des))
dtm = DocumentTermMatrix(corpus, control = list(wordLengths=c(1,Inf)))

# Select the most frequent terms
frequent = findFreqTerms(dtm, 10000)
frequent

# tfidf (natural tf + idf + no normalizaition)
tfidf = weightSMART(dtm, spec='ntn')

# Join description table with tfidf by the listing id
tfidf_df = as.data.frame(inspect(tfidf[,which(tfidf$dimnames$Terms%in%frequent)]))
colnames(tfidf_df) = sapply(colnames(tfidf_df), function(v) paste('term',v,sep='_'), USE.NAMES = FALSE)
tfidf_df['listing_id'] = non_empty_clean_des$listing_id

tfidf_df_full=description[c('listing_id', 'interest_level', 'interest_Nbr', 'clean_wordcount')]
tfidf_df_full['wordcount'] = tfidf_df_full$clean_wordcount
tfidf_df_full$clean_wordcount=NULL

tfidf_df_full = merge(x = tfidf_df_full, y = tfidf_df, by = "listing_id", all.x = TRUE)

# Convert all na to 0
tfidf_df_full[is.na(tfidf_df_full)] = 0

write.csv(tfidf_df_full, file="train_description_wordcount_tfidf.csv", row.names=FALSE)

####################
## Sentiment extraction
#install.packages('syuzhet')
#install.packages('DT')
library(syuzhet)
library(DT)
sentiment = get_nrc_sentiment(description$origin_des)
senti = sapply(colnames(sentiment), function(v) paste('senti',v,sep='_'), USE.NAMES = FALSE)
colnames(sentiment) = senti
head(sentiment)

sentiment['listing_id'] = description$listing_id
sentiment['interest_level'] = description$interest_level
sentiment['interest_Nbr'] = description$interest_Nbr

sentiment = sentiment[c('listing_id', 'interest_level', 'interest_Nbr', senti)]

write.csv(sentiment, "../processed_data/train_description_sentiment.csv", row.names=FALSE)
