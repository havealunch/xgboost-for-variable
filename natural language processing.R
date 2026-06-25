# Load required libraries
library(tidyverse)
library(text2vec)
library(bert)
library(topicmodels)
library(tm)
library(SnowballC)
library(wordcloud)
library(LDAvis)
library(factoextra)
library(psych)
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(jiebaR)  # For Chinese text segmentation

# Set seed for reproducibility
set.seed(20241021)

#===============================================================================
# 1. LOAD AND PREPARE DATA
#===============================================================================

# Assuming you have a CSV file with Weibo data
# Expected columns: 
#   - city: City name
#   - weibo_id: Unique ID for each post
#   - content: Weibo text content
#   - post_time: Timestamp of the post
#   - user_id: User ID (optional)
#   - likes: Number of likes (optional)
#   - comments: Number of comments (optional)

# Load the data
weibo_data <- read.csv("weibo_flood_data.csv", stringsAsFactors = FALSE)

# Preview the data
str(weibo_data)
head(weibo_data)

# Data cleaning function
clean_weibo_text <- function(text) {
  # Remove URLs
  text <- gsub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+", "", text)
  # Remove mentions (@username)
  text <- gsub("@[\\w_]+", "", text)
  # Remove hashtags (#topic#)
  text <- gsub("#[^#]+#", "", text)
  # Remove emojis and special characters (keep Chinese characters)
  text <- gsub("[^\u4e00-\u9fa5\\s]", "", text)
  # Remove extra spaces
  text <- gsub("\\s+", " ", text)
  text <- trimws(text)
  return(text)
}

# Apply cleaning
weibo_data$content_clean <- sapply(weibo_data$content, clean_weibo_text)

# Filter to posts within one month of the disaster
# Assuming the flood event started on July 1, 2020
disaster_date <- as.Date("2020-07-01")
weibo_data$post_date <- as.Date(weibo_data$post_time)
weibo_data$days_after_disaster <- as.numeric(weibo_data$post_date - disaster_date)

# Keep only posts within 30 days after the disaster
weibo_data_filtered <- weibo_data %>%
  filter(days_after_disaster >= 0 & days_after_disaster <= 30)

cat("Number of posts after filtering:", nrow(weibo_data_filtered), "\n")

#===============================================================================
# 2. SENTIMENT ANALYSIS USING BERT (Pre-trained Chinese BERT)
#===============================================================================

# Option 1: Using the bert package (if available)
# Install: remotes::install_github("rstudio/bert")

# Function to predict sentiment using BERT (simplified version)
# In practice, you would use a pre-trained Chinese BERT model
# This is a placeholder - you'll need to implement actual BERT prediction

predict_bert_sentiment <- function(texts) {
  # This is a simplified placeholder
  # In practice, you would use a trained BERT model
  # Or use an API service for sentiment analysis
  
  # For demonstration, we'll create a simple rule-based sentiment
  # Real implementation should use actual BERT model
  
  # Load sentiment dictionaries (simplified)
  positive_words <- c("加油", "坚持", "帮助", "支持", "感谢", "好转", "恢复", "重建", 
                      "安全", "救援", "温暖", "感动", "团结", "希望", "振奋", "坚强")
  
  negative_words <- c("损失", "困难", "严重", "破坏", "死亡", "失踪", "洪水", "淹没", 
                      "倒塌", "危险", "恐惧", "焦虑", "痛苦", "艰难", "压力", "崩溃")
  
  sentiments <- sapply(texts, function(text) {
    # Count positive and negative words
    pos_count <- sum(str_count(text, positive_words))
    neg_count <- sum(str_count(text, negative_words))
    
    if (pos_count > neg_count) {
      return("positive")
    } else if (neg_count > pos_count) {
      return("negative")
    } else {
      return("neutral")
    }
  })
  
  return(sentiments)
}

# Apply sentiment analysis
weibo_data_filtered$sentiment <- predict_bert_sentiment(weibo_data_filtered$content_clean)

# Calculate sentiment distribution by city
city_sentiment <- weibo_data_filtered %>%
  group_by(city) %>%
  summarise(
    total_posts = n(),
    positive_count = sum(sentiment == "positive"),
    negative_count = sum(sentiment == "negative"),
    neutral_count = sum(sentiment == "neutral"),
    positive_ratio = positive_count / total_posts,
    negative_ratio = negative_count / total_posts,
    neutral_ratio = neutral_count / total_posts,
    .groups = 'drop'
  )

print(head(city_sentiment))

#===============================================================================
# 3. TOPIC MODELING USING LDA
#===============================================================================

# Function for Chinese text segmentation
segment_chinese_text <- function(texts) {
  # Initialize jieba
  seg <- worker(type = "tag", dict = "dict")
  
  # Segment each text
  segmented <- lapply(texts, function(text) {
    words <- segment(text, seg)
    # Remove single-character words and stopwords
    words <- words[nchar(words) > 1]
    return(words)
  })
  
  return(segmented)
}

# Prepare text for LDA
# Create a corpus
corpus <- Corpus(VectorSource(weibo_data_filtered$content_clean))

# Chinese stopwords (simplified list - you should use a comprehensive list)
chinese_stopwords <- c("的", "了", "在", "是", "我", "有", "和", "就", "不", "人", 
                       "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", 
                       "你", "会", "着", "没有", "看", "好", "自己", "这", "那", 
                       "它", "他", "她", "们", "与", "或", "但", "因为", "所以", 
                       "如果", "虽然", "然而", "而且", "但是", "并且")

# Clean text
clean_corpus <- tm_map(corpus, content_transformer(tolower))
clean_corpus <- tm_map(clean_corpus, content_transformer(removePunctuation))
clean_corpus <- tm_map(clean_corpus, content_transformer(removeNumbers))
clean_corpus <- tm_map(clean_corpus, removeWords, chinese_stopwords)
clean_corpus <- tm_map(clean_corpus, stripWhitespace)

# Create Document-Term Matrix
dtm <- DocumentTermMatrix(clean_corpus)

# Remove terms that appear in less than 1% of documents
dtm <- dtm[, terms(dtm)[colSums(as.matrix(dtm)) > 0.01 * nrow(dtm)]]

# Remove empty documents
rowTotals <- rowSums(as.matrix(dtm))
dtm <- dtm[rowTotals > 0, ]

# Perform LDA topic modeling
# Determine optimal number of topics using perplexity
topic_numbers <- seq(5, 15, by = 5)
perplexity_scores <- sapply(topic_numbers, function(k) {
  lda_model <- LDA(dtm, k = k, method = "Gibbs", 
                   control = list(iter = 500, burnin = 200, thin = 50))
  return(perplexity(lda_model))
})

# Choose optimal k (lower perplexity is better)
optimal_k <- topic_numbers[which.min(perplexity_scores)]
cat("Optimal number of topics:", optimal_k, "\n")

# Fit final LDA model
lda_model <- LDA(dtm, k = optimal_k, method = "Gibbs", 
                 control = list(iter = 1000, burnin = 500, thin = 100))

# Extract topic probabilities for each document
topic_distribution <- posterior(lda_model)$topics
colnames(topic_distribution) <- paste0("Topic", 1:optimal_k)

# Label topics based on top words
get_topic_labels <- function(model, n_terms = 10) {
  terms <- terms(model, n_terms)
  labels <- apply(terms, 2, function(topic_terms) {
    paste(topic_terms[1:5], collapse = ", ")
  })
  return(labels)
}

topic_labels <- get_topic_labels(lda_model)

# Categorize topics into disruption types
# This requires manual labeling based on the top words
topic_categories <- data.frame(
  Topic = paste0("Topic", 1:optimal_k),
  Top_Terms = topic_labels,
  Category = NA,  # To be filled manually
  Disruption_Type = NA  # education/work/transportation/other
)

# Print top terms for each topic to help with labeling
for(i in 1:optimal_k) {
  cat("\n", topic_categories$Topic[i], "\n")
  cat("Top terms:", topic_categories$Top_Terms[i], "\n")
}

# Example categorization (you need to adjust based on your actual topics)
# topic_categories$Category[1] <- "Education Disruption"
# topic_categories$Disruption_Type[1] <- "education"
# ... etc.

# Calculate topic distribution by city
city_topics <- weibo_data_filtered %>%
  mutate(doc_id = 1:n()) %>%
  bind_cols(as.data.frame(topic_distribution)) %>%
  group_by(city) %>%
  summarise(
    across(starts_with("Topic"), mean, .names = "mean_{.col}"),
    education_disruption_ratio = mean(ifelse(topic_categories$Disruption_Type == "education", 
                                             topic_distribution[, which(topic_categories$Disruption_Type == "education")], 0)),
    work_disruption_ratio = mean(ifelse(topic_categories$Disruption_Type == "work", 
                                        topic_distribution[, which(topic_categories$Disruption_Type == "work")], 0)),
    transport_disruption_ratio = mean(ifelse(topic_categories$Disruption_Type == "transportation", 
                                             topic_distribution[, which(topic_categories$Disruption_Type == "transportation")], 0)),
    .groups = 'drop'
  )

print(head(city_topics))

#===============================================================================
# 4. CALCULATE DISASTER INTERRUPTION INDEX
#===============================================================================

# Set weights based on research needs
alpha <- 0.4  # Weight for negative sentiment
beta <- 0.4   # Weight for education/work disruption
gamma <- 0.2  # Weight for other topics

# Merge sentiment and topic data
city_index <- city_sentiment %>%
  left_join(city_topics, by = "city")

# Calculate interruption index
city_index <- city_index %>%
  mutate(
    # If education_disruption_ratio and work_disruption_ratio are not available,
    # use the combined topic scores
    education_work_disruption = ifelse(
      "education_disruption_ratio" %in% names(city_index) & 
        "work_disruption_ratio" %in% names(city_index),
      (education_disruption_ratio + work_disruption_ratio) / 2,
      rowMeans(select(., starts_with("mean_Topic")), na.rm = TRUE)
    ),
    
    # Calculate other topics ratio
    other_topics_ratio = rowMeans(select(., starts_with("mean_Topic")), na.rm = TRUE) - 
      ifelse("education_disruption_ratio" %in% names(city_index), 
             education_disruption_ratio + work_disruption_ratio, 0),
    
    # Calculate the comprehensive index
    interruption_index = alpha * negative_ratio + 
      beta * education_work_disruption + 
      gamma * other_topics_ratio
  )

print(head(city_index))

#===============================================================================
# 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
#===============================================================================

# Prepare data for PCA
pca_vars <- c("negative_ratio", "positive_ratio", "neutral_ratio", "interruption_index")

# Add topic-specific variables if available
topic_vars <- grep("mean_Topic", names(city_index), value = TRUE)
pca_vars <- c(pca_vars, topic_vars)

# Select only complete cases
pca_data <- city_index %>%
  select(all_of(pca_vars)) %>%
  na.omit()

# Standardize data
pca_data_scaled <- scale(pca_data)

# Perform PCA
pca_result <- prcomp(pca_data_scaled, center = TRUE, scale. = TRUE)

# Summary of PCA
summary(pca_result)

# Get eigenvalues
eigenvalues <- pca_result$sdev^2
variance_explained <- eigenvalues / sum(eigenvalues) * 100

# Create scree plot
scree_plot <- data.frame(
  PC = 1:length(eigenvalues),
  Eigenvalue = eigenvalues,
  Variance = variance_explained
) %>%
  ggplot(aes(x = PC, y = Eigenvalue)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_line(color = "red", size = 1) +
  geom_point(color = "red", size = 3) +
  theme_minimal() +
  labs(title = "Scree Plot of PCA",
       x = "Principal Component",
       y = "Eigenvalue")

print(scree_plot)

# Get loadings
loadings <- pca_result$rotation
print("PCA Loadings:")
print(loadings[, 1:5])  # First 5 components

# Get variable contributions to principal components
contributions <- data.frame(
  Variable = rownames(loadings),
  PC1 = loadings[, 1],
  PC2 = loadings[, 2],
  PC3 = loadings[, 3]
)

print(contributions)

# Biplot
biplot(pca_result, main = "PCA Biplot of City Indicators")

# Visualize PCA results
pca_scores <- as.data.frame(pca_result$x)
pca_scores$city <- city_index$city[1:nrow(pca_scores)]

# PCA score plot
pca_plot <- pca_scores %>%
  ggplot(aes(x = PC1, y = PC2, label = city)) +
  geom_point(size = 3, color = "steelblue") +
  geom_text(vjust = -0.5, size = 3) +
  theme_minimal() +
  labs(title = "PCA Score Plot",
       x = paste0("PC1 (", round(variance_explained[1], 1), "%)"),
       y = paste0("PC2 (", round(variance_explained[2], 1), "%)")) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5)

print(pca_plot)

#===============================================================================
# 6. WEIGHT CALCULATION FOR PCA COMPONENTS
#===============================================================================

# Calculate weights for each feature in principal components
weight_coefficients <- data.frame(
  Feature = rownames(loadings),
  PC1_Weight = abs(loadings[, 1]) / sum(abs(loadings[, 1])),
  PC2_Weight = abs(loadings[, 2]) / sum(abs(loadings[, 2])),
  PC3_Weight = abs(loadings[, 3]) / sum(abs(loadings[, 3]))
)

print("Weight coefficients for each feature:")
print(head(weight_coefficients, 10))

# Identify top contributing features for each PC
top_features_pc1 <- weight_coefficients %>%
  arrange(desc(PC1_Weight)) %>%
  head(5)

top_features_pc2 <- weight_coefficients %>%
  arrange(desc(PC2_Weight)) %>%
  head(5)

top_features_pc3 <- weight_coefficients %>%
  arrange(desc(PC3_Weight)) %>%
  head(5)

cat("\nTop 5 features contributing to PC1:\n")
print(top_features_pc1)

cat("\nTop 5 features contributing to PC2:\n")
print(top_features_pc2)

cat("\nTop 5 features contributing to PC3:\n")
print(top_features_pc3)

#===============================================================================
# 7. EXPORT RESULTS
#===============================================================================

# Export city interruption indices
write.csv(city_index, "city_disaster_interruption_index.csv", row.names = FALSE)

# Export PCA results
pca_results <- list(
  pca_object = pca_result,
  loadings = loadings,
  variance_explained = variance_explained,
  scores = pca_scores,
  weight_coefficients = weight_coefficients
)

saveRDS(pca_results, "pca_analysis_results.rds")

# Export topic model results
topic_results <- list(
  lda_model = lda_model,
  topic_distribution = topic_distribution,
  topic_labels = topic_labels,
  optimal_k = optimal_k,
  topic_categories = topic_categories
)

saveRDS(topic_results, "topic_model_results.rds")

# Create summary report
summary_report <- list(
  total_posts = nrow(weibo_data_filtered),
  unique_cities = length(unique(weibo_data_filtered$city)),
  sentiment_summary = city_sentiment %>%
    summarise(
      avg_positive = mean(positive_ratio),
      avg_negative = mean(negative_ratio),
      avg_neutral = mean(neutral_ratio)
    ),
  optimal_topics = optimal_k,
  pca_variance_explained = variance_explained[1:5],
  top_features_pc1 = top_features_pc1
)

# Save summary report
capture.output(
  print(summary_report),
  file = "nlp_analysis_summary.txt"
)

cat("Analysis complete! Results saved to:\n")
cat("1. city_disaster_interruption_index.csv\n")
cat("2. pca_analysis_results.rds\n")
cat("3. topic_model_results.rds\n")
cat("4. nlp_analysis_summary.txt\n")

#===============================================================================
# 8. VISUALIZATION FUNCTIONS
#===============================================================================

# Create a comprehensive visualization of results
plot_sentiment_distribution <- function(data) {
  data %>%
    select(city, positive_ratio, negative_ratio, neutral_ratio) %>%
    pivot_longer(cols = -city, names_to = "sentiment", values_to = "ratio") %>%
    ggplot(aes(x = city, y = ratio, fill = sentiment)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Sentiment Distribution by City",
         x = "City", y = "Proportion")
}

# Plot interruption index by city
plot_interruption_index <- function(data) {
  data %>%
    ggplot(aes(x = reorder(city, interruption_index), y = interruption_index)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Disaster Interruption Index by City",
         x = "City", y = "Interruption Index")
}

# Generate plots
sentiment_plot <- plot_sentiment_distribution(city_index)
interruption_plot <- plot_interruption_index(city_index)

print(sentiment_plot)
print(interruption_plot)

# Save plots
ggsave("sentiment_distribution.png", sentiment_plot, width = 10, height = 6)
ggsave("interruption_index.png", interruption_plot, width = 10, height = 6)