transfer=list('low'=1, 'medium'=2, 'high'=3)

####################
# # Read from train data set
library(jsonlite)
train_data<-fromJSON('../data/train.json')
train_data <- lapply(train_data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x,use.names = FALSE)
})
test_data<-fromJSON('../data/test.json')
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

write.csv(description, file= "../processed_data/train_description.csv", row.names=FALSE)

####################
# Add this column to the baseline data
train = read.csv('../processed_data/train_baseline11_v3.csv')
train = merge(train, description[c('listing_id', 'clean_wordcount')])
write.csv(train, file = '../processed_data/train_baseline12_v4.csv', row.names = FALSE)

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

tfidf_df_full=description[c('listing_id', 'interest_level', 'interest_Nbr')]
tfidf_df_full$clean_wordcount=NULL

tfidf_df_full = merge(x = tfidf_df_full, y = tfidf_df, by = "listing_id", all.x = TRUE)

# Convert all na to 0
tfidf_df_full[is.na(tfidf_df_full)] = 0

write.csv(tfidf_df_full, file="../processed_data/train_description_tfidf.csv", row.names=FALSE)

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

####################
# Word cloud for each interest level and munally select the terms
description = read.csv('../processed_data/train_description.csv')
non_empty_clean_des = description[which(description$clean_wordcount!=0), ]

low = non_empty_clean_des[non_empty_clean_des$interest_level=='low',]
medium = non_empty_clean_des[non_empty_clean_des$interest_level=='medium', ]
high = non_empty_clean_des[non_empty_clean_des$interest_level=='high', ]

# Build the corpus for the terms in each interest level
#install.packages('wordcloud')
library(wordcloud)
library(tm)

lowCorpus = VCorpus(VectorSource(low$clean_des))
mediumCorpus = VCorpus(VectorSource(medium$clean_des))
highCorpus = VCorpus(VectorSource(high$clean_des))

# Plot the word cloud for all frequent temrs in each interst level
#install.packages("RColorBrewer")
library(RColorBrewer)
pal2 <- brewer.pal(8,"Dark2")
wordcloud(lowCorpus, max.words = 50, random.order = T, colors = pal2)
wordcloud(mediumCorpus, max.words = 50, random.order = T, colors = pal2)
wordcloud(highCorpus, max.words = 50, random.order = T, colors = pal2)
# The words in each level contain too many common things, we want to find the words specifically 
# represents each level

# # Find the frequecy of terms in each interst level
# lowDTM = DocumentTermMatrix(lowCorpus, control = list(wordLengths=c(1,Inf)))
# mediumDTM = DocumentTermMatrix(mediumCorpus, control = list(wordLengths=c(1,Inf)))
# highDTM = DocumentTermMatrix(highCorpus, control = list(wordLengths=c(1,Inf)))
#
# lowFrequent = apply(lowDTM, 2,sum)
# length(lowFrequent)
# # 26869
# mediumFrequent = apply(mediumDTM, 2,sum)
# length(mediumFrequent)
# # 15673
# highFrequent = apply(highDTM, 2,sum)
# length(highFrequent)
# # 8494

# # Select the most frequent terms
# lowFrequent = findFreqTerms(lowDTM)
# mediumFrequent = findFreqTerms(mediumDTM)
# highFrequent = findFreqTerms(highDTM)

# lowTerms = names(sort(lowFrequent, decreasing = T)[1:500])
# mediumTerms = names(sort(mediumFrequent, decreasing = T)[1:500])
# highTerms = names(sort(highFrequent, decreasing = T)[1:500])

# # no significant findings for the terms used particularly in each level
# setdiff(lowTerms, union(mediumTerms, highTerms))
# setdiff(mediumTerms, union(lowTerms, highTerms))
# setdiff(highTerms, union(mediumTerms, lowTerms))

# # Trying to remove the terms that exist in all levels 
# commonTerms = intersect(names(lowFrequent), names(mediumFrequent))
# commonTerms = intersect(commonTerms, names(highFrequent))
# length(commonTerms)
# # 6306

# lowTerms = lowFrequent[setdiff(names(lowFrequent), commonTerms)]
# d <- data.frame(word = names(lowTerms),freq=lowTerms)
# # plot the word cloud for each level again
# wordcloud(d$word, d$freq, max.words = 50, random.order = T, colors = pal2)
# # However, after remove the common words in each level, the remaining terms appear to be
# # very weird, not like English words and the frequency is too low.


# Trying to remove the terms that appear in xx.x% percent of the documents
# http://stackoverflow.com/questions/25905144/removing-overly-common-words-occur-in-more-than-80-of-the-documents-in-r
corpus = VCorpus(VectorSource(non_empty_clean_des$clean_des))
dtm = DocumentTermMatrix(corpus, control = list(wordLengths=c(1,Inf)))
removeCommonTerms <- function (x, pct) {
  stopifnot(inherits(x, c("DocumentTermMatrix", "TermDocumentMatrix")), 
            is.numeric(pct), pct > 0, pct < 1)
  m <- if (inherits(x, "TermDocumentMatrix")) 
    t(x)
  else x
  t <- table(m$j) < m$nrow * (pct)
  termIndex <- as.numeric(names(t[t]))
  if (inherits(x, "TermDocumentMatrix")) 
    return(x[termIndex, ])
  else return(x[,termIndex])
}
removed_dtm = removeCommonTerms(dtm, 0.25)

lowDTM2 = removed_dtm[non_empty_clean_des$interest_level=='low', ]
lowFrequent2 = sort(colSums(as.matrix(lowDTM2)), decreasing = TRUE)
png("wordcloud_low.png", width=1280,height=800)
d <- data.frame(word = names(lowFrequent2),freq=lowFrequent2)
wordcloud(d$word, d$freq, max.words = 50, scale=c(3,.1), random.order = FALSE, colors = pal2)
dev.off()

mediumDTM2 = removed_dtm[non_empty_clean_des$interest_level=='medium', ]
mediumFrequent2 = sort(colSums(as.matrix(mediumDTM2)), decreasing = TRUE)
png("wordcloud_medium.png", width=1280,height=800)
d <- data.frame(word = names(mediumFrequent2), freq=mediumFrequent2)
wordcloud(d$word, d$freq, max.words = 50, scale=c(3,.2), random.order = FALSE, colors = pal2)
dev.off()

highDTM2 = removed_dtm[non_empty_clean_des$interest_level=='high', ]
highFrequent2 = sort(colSums(as.matrix(highDTM2)), decreasing = TRUE)
png("wordcloud_high.png", width=1280,height=800)
d <- data.frame(word = names(highFrequent2),freq=highFrequent2)
wordcloud(d$word, d$freq, max.words = 50, scale=c(3,.2), random.order = FALSE, colors = pal2)
dev.off()

# Check the most frequent term set for each level and get the joint of the sets
lowTerms2 = names(lowFrequent2[1:50])
mediumTerms2 = names(mediumFrequent2[1:50])
highTerms2 = names(highFrequent2[1:50])

setdiff(lowTerms2, union(mediumTerms2, highTerms2))
# [1] "color"        "income"       "enough"       "note"         "electric"    
# [6] "tenant"       "value"        "entertaining" "fi"           "wi"          
# [11] "asap"         "deep"         "gardens"      "heating"      "hotel"       
# [16] "astoria"      "harlem"       "ceramic"      "couch"        "date"        
# [21] "priced"       "prices"       "tables"       "approval"     "commute"     
# [26] "dresser"      "etc"          "immediately"  "second"       "trendy"      
# [31] "way"          "front"        "personal"    
setdiff(mediumTerms2, union(lowTerms2, highTerms2))
# [1] "dryer"      "apartments" "west"       "fitness"    "schedule"   "deck"
setdiff(highTerms2, union(mediumTerms2, lowTerms2))
# [1] "rent"     "super"    "included" "will"     "street"   "bed"      "two"     
# [8] "away"  

union = union(lowTerms2, mediumTerms2)
union = union(temp, highTerms2)
length(union)
# [1] 65

# tfidf (natural tf + idf + no normalizaition)
tfidf = weightSMART(dtm, spec='ntn')

# Join description table with tfidf by the listing id
tfidf_df = as.data.frame(inspect(tfidf[,which(tfidf$dimnames$Terms %in% union)]))
colnames(tfidf_df) = sapply(colnames(tfidf_df), function(v) paste('term',v,sep='_'), USE.NAMES = FALSE)
tfidf_df['listing_id'] = non_empty_clean_des$listing_id

tfidf_df_full=description[c('listing_id', 'interest_level', 'interest_Nbr', 'clean_wordcount')]
tfidf_df_full['wordcount'] = tfidf_df_full$clean_wordcount
tfidf_df_full$clean_wordcount=NULL

tfidf_df_full = merge(x = tfidf_df_full, y = tfidf_df, by = "listing_id", all.x = TRUE)

# Convert all na to 0
tfidf_df_full[is.na(tfidf_df_full)] = 0

write.csv(tfidf_df_full, file="../processed_data/train_description_wordcount_tfidf_new.csv", row.names=FALSE)
