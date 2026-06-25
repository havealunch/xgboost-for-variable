# Install packages
install.packages("xgboost")
install.packages("tidyverse")
install.packages("skimr")
install.packages("DataExplorer")
install.packages("caret")
install.packages("pROC")
install.packages("shapviz")
install.packages("iml")
install.packages("shapr")
install.packages("GGally")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("plotly")
install.packages("gridExtra")

# Load libraries
library(xgboost)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(caret)
library(shapviz)
library(pROC)
library(iml)
library(shapr)
library(GGally)
library(reshape2)
library(ggplot2)
library(dplyr)
library(plotly)
library(gridExtra)

# Load data
campus <- read.csv(file.choose())

# Check column names
print(colnames(campus))

# Identify target variable (Climate Inequality Coefficient)
# Assuming the target variable is named "WE" or similar
target_var <- "WE"  # Change this to your actual target variable name

# Check data structure
str(campus)
summary(campus)

# Statistical summary for numeric columns
for(col in names(campus)) {
  if(is.numeric(campus[[col]])) {
    cat("\nColumn:", col, "\n")
    cat("Max:", max(campus[[col]], na.rm = TRUE), "\n")
    cat("Min:", min(campus[[col]], na.rm = TRUE), "\n")
    cat("Mean:", mean(campus[[col]], na.rm = TRUE), "\n")
    
    if(length(unique(campus[[col]])) > 1) {
      cat("SD:", sd(campus[[col]], na.rm = TRUE), "\n")
      cat("Var:", var(campus[[col]], na.rm = TRUE), "\n")
    } else {
      cat("SD and Var cannot be calculated - all values are identical or empty.\n")
    }
  } else {
    cat("\nColumn:", col, "is not numeric, skipping.\n")
  }
}

# Data overview
skimr::skim(campus)

# Missing data visualization
plot_missing(campus)

# Target variable distribution
hist(campus[[target_var]], breaks = 20, main = paste("Distribution of", target_var), 
     xlab = target_var, col = "steelblue")

# Correlation analysis to identify key variables
numeric_vars <- campus %>% select(where(is.numeric))
cor_matrix <- cor(numeric_vars, use = "complete.obs")
cor_with_target <- cor_matrix[target_var, ]
cor_with_target <- sort(cor_with_target[!names(cor_with_target) %in% target_var], decreasing = TRUE)
print("Top 10 variables correlated with climate inequality coefficient:")
print(head(cor_with_target, 10))

# Split data
set.seed(20241021)
trains <- createDataPartition(y = campus[[target_var]], p = 0.85, list = FALSE, times = 1)
trains2 <- sample(trains, nrow(campus) * 0.8)
valids <- setdiff(trains, trains2)

data_train <- campus[trains2, ]
data_valid <- campus[valids, ]
data_test <- campus[-trains, ]

# Check target variable distribution in splits
par(mfrow = c(1, 3))
hist(data_train[[target_var]], breaks = 20, main = "Train", xlab = target_var, col = "lightblue")
hist(data_valid[[target_var]], breaks = 20, main = "Validation", xlab = target_var, col = "lightgreen")
hist(data_test[[target_var]], breaks = 20, main = "Test", xlab = target_var, col = "lightpink")
par(mfrow = c(1, 1))

# Data preparation - identify feature columns
# Exclude target variable and any ID columns
feature_cols <- setdiff(colnames(campus), c(target_var, "uid", "ID", "id"))
print("Feature columns:")
print(feature_cols)

# Prepare data for modeling
dvfunc <- dummyVars(~ ., data = data_train[, feature_cols, drop = FALSE], fullRank = TRUE)

data_trainx <- predict(dvfunc, newdata = data_train[, feature_cols, drop = FALSE])
data_trainy <- data_train[[target_var]]

data_validx <- predict(dvfunc, newdata = data_valid[, feature_cols, drop = FALSE])
data_validy <- data_valid[[target_var]]

data_testx <- predict(dvfunc, newdata = data_test[, feature_cols, drop = FALSE])
data_testy <- data_test[[target_var]]

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = data_trainx, label = data_trainy)
dvalid <- xgb.DMatrix(data = data_validx, label = data_validy)
dtest <- xgb.DMatrix(data = data_testx, label = data_testy)

# Watchlist
watchlist <- list(train = dtrain, test = dvalid)

# Train initial model
fit_xgb_reg <- xgb.train(
  data = dtrain,
  eta = 0.1,
  gamma = 0.001,
  max_depth = 8,
  subsample = 1,
  colsample_bytree = 0.5,
  objective = "reg:squarederror",
  nrounds = 1500,
  watchlist = watchlist,
  verbose = 1,
  min_child_weight = 3,
  print_every_n = 100,
  early_stopping_rounds = 400
)

# Model summary
print(fit_xgb_reg)

# Feature importance from XGBoost
importance_matrix <- xgb.importance(model = fit_xgb_reg)
print(importance_matrix[1:15, ])

# Plot feature importance
xgb.plot.importance(importance_matrix[1:15, ])

# Predictions
trainpred <- predict(fit_xgb_reg, newdata = dtrain)
testpred <- predict(fit_xgb_reg, newdata = dtest)

# Training set metrics
train_metrics <- defaultSummary(data.frame(obs = data_trainy, pred = trainpred))
print("Training set metrics:")
print(train_metrics)

train_mse <- mse(data_trainy, trainpred)
train_rmse <- sqrt(train_mse)
train_mae <- MAE(data_trainy, trainpred)
cat("Training MSE:", train_mse, "\n")
cat("Training RMSE:", train_rmse, "\n")
cat("Training MAE:", train_mae, "\n")
cat("Training R-squared:", cor(data_trainy, trainpred)^2, "\n")

# Test set metrics
test_metrics <- defaultSummary(data.frame(obs = data_testy, pred = testpred))
print("Test set metrics:")
print(test_metrics)

test_mse <- mse(data_testy, testpred)
test_rmse <- sqrt(test_mse)
test_mae <- MAE(data_testy, testpred)
cat("Test MSE:", test_mse, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test MAE:", test_mae, "\n")
cat("Test R-squared:", cor(data_testy, testpred)^2, "\n")

# Visualization of predictions
par(mfrow = c(1, 2))

# Training set
plot(data_trainy, trainpred, 
     xlab = "Actual", ylab = "Prediction", 
     main = "XGBoost - Training Set",
     col = "blue", pch = 16)
abline(a = 0, b = 1, col = "red", lwd = 2)
trainlinmod <- lm(trainpred ~ data_trainy)
abline(trainlinmod, col = "green", lwd = 2)
legend("topleft", legend = c("Ideal", "Model"), 
       col = c("red", "green"), lwd = 2)

# Test set
plot(data_testy, testpred, 
     xlab = "Actual", ylab = "Prediction", 
     main = "XGBoost - Test Set",
     col = "blue", pch = 16)
abline(a = 0, b = 1, col = "red", lwd = 2)
testlinmod <- lm(testpred ~ data_testy)
abline(testlinmod, col = "green", lwd = 2)
legend("topleft", legend = c("Ideal", "Model"), 
       col = c("red", "green"), lwd = 2)

par(mfrow = c(1, 1))

# Hyperparameter tuning
tune_grid <- expand.grid(
  nrounds = c(500, 1000, 1500),
  max_depth = c(3, 4, 5, 6),
  eta = c(0.05, 0.1, 0.2),
  gamma = c(0, 0.001, 0.01, 0.1),
  colsample_bytree = c(0.3, 0.5, 0.7),
  subsample = c(0.5, 0.7, 1),
  min_child_weight = c(1, 3, 5)
)

# Use smaller grid for faster computation
tune_grid_small <- expand.grid(
  nrounds = c(500, 1000),
  max_depth = c(4, 6),
  eta = c(0.1, 0.3),
  gamma = c(0, 0.01),
  colsample_bytree = c(0.5, 0.7),
  subsample = c(0.7, 1),
  min_child_weight = c(1, 3)
)

train_control <- trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE,
  allowParallel = TRUE,
  savePredictions = "final"
)

# Run tuning (using smaller grid for speed)
xgb_tune <- tryCatch({
  train(
    x = data_validx,
    y = data_validy,
    method = "xgbTree",
    trControl = train_control,
    tuneGrid = tune_grid_small
  )
}, error = function(e) {
  message("Tuning error: ", e$message)
  # Fallback: use default parameters
  train(
    x = data_validx,
    y = data_validy,
    method = "xgbTree",
    trControl = train_control
  )
})

print("Best tuning parameters:")
print(xgb_tune$bestTune)

# Train final model with best parameters
best_params <- xgb_tune$bestTune
final_model <- xgb.train(
  data = dtrain,
  eta = best_params$eta,
  gamma = 0.001,
  max_depth = best_params$max_depth,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  objective = "reg:squarederror",
  nrounds = best_params$nrounds,
  watchlist = watchlist,
  verbose = 0,
  min_child_weight = best_params$min_child_weight
)

# Calculate SHAP values
shap_xgboost <- shapviz(final_model, X_pred = as.matrix(data_trainx))
print("SHAP object created successfully")

# SHAP Summary Plot
sv_importance(shap_xgboost, kind = "beeswarm", max_display = 20, 
              title = "SHAP Variable Importance - Top 20 Features")

# SHAP Bar Plot
sv_importance(shap_xgboost, kind = "bar", max_display = 20,
              title = "SHAP Feature Importance - Top 20 Features")

# SHAP Variable Importance with percentages
shap_values_matrix <- as.matrix(shap_xgboost$S)
shap_importance <- colSums(abs(shap_values_matrix))
shap_percentage <- 100 * shap_importance / sum(shap_importance)

shap_df <- data.frame(
  Variable = names(shap_importance),
  Importance = shap_importance,
  Percentage = shap_percentage
)

# Sort by importance
shap_df <- shap_df[order(-shap_df$Importance), ]
shap_df$Variable <- factor(shap_df$Variable, levels = rev(shap_df$Variable))

# Identify key variables (top 10)
key_variables <- head(shap_df, 10)
print("Top 10 key variables affecting climate inequality coefficient:")
print(key_variables)

# Plot SHAP variable importance
ggplot(shap_df[1:20, ], aes(x = Variable, y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "SHAP Variable Importance",
       x = "Variable",
       y = "Mean |SHAP Value|") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), 
            hjust = -0.1, 
            size = 3) +
  theme_minimal()

# SHAP Summary Plot with color
sv_importance(shap_xgboost, kind = "beeswarm", max_display = 20) +
  labs(title = "SHAP Summary Plot")

# SHAP Dependence Plots for top 5 important variables
top5_vars <- head(shap_df$Variable, 5)

# Create dependence plots for top variables
for(var_name in top5_vars) {
  print(sv_dependence(shap_xgboost, v = var_name, color_var = "auto") +
          labs(title = paste("SHAP Dependence Plot:", var_name)))
}

# SHAP Force Plot for a specific observation
sv_force(shap_xgboost, row_id = 1)

# Calculate contribution of each variable for individual predictions
shap_contrib <- as.matrix(shap_xgboost$S)
colnames(shap_contrib) <- colnames(data_trainx)

# Identify the most important variable for each observation
max_contrib_indices <- apply(abs(shap_contrib), 1, which.max)
max_contrib_vars <- colnames(shap_contrib)[max_contrib_indices]
max_contrib_values <- shap_contrib[cbind(1:nrow(shap_contrib), max_contrib_indices)]

# Create summary dataframe with key variables identified
summary_df <- data.frame(
  Observation = 1:nrow(shap_contrib),
  Total_SHAP = rowSums(shap_contrib),
  Key_Variable = max_contrib_vars,
  Key_Contribution = max_contrib_values
)

head(summary_df)

# Export key variables and SHAP values
write.csv(shap_df, "shap_variable_importance.csv", row.names = FALSE)
write.csv(summary_df, "shap_individual_contributions.csv", row.names = FALSE)

# Save the final model
xgb.save(final_model, "climate_inequality_model.model")

# Create comprehensive summary report
cat("\n========================================\n")
cat("KEY VARIABLES IDENTIFIED\n")
cat("========================================\n")
print(key_variables)

cat("\n========================================\n")
cat("MODEL PERFORMANCE SUMMARY\n")
cat("========================================\n")
cat("Training R-squared:", cor(data_trainy, trainpred)^2, "\n")
cat("Test R-squared:", cor(data_testy, testpred)^2, "\n")
cat("Training RMSE:", sqrt(mean((data_trainy - trainpred)^2)), "\n")
cat("Test RMSE:", sqrt(mean((data_testy - testpred)^2)), "\n")

# Create a complete SHAP analysis report
shap_report <- list(
  model = final_model,
  shap_values = shap_xgboost,
  variable_importance = shap_df,
  key_variables = key_variables,
  model_performance = data.frame(
    Dataset = c("Training", "Test"),
    RMSE = c(sqrt(mean((data_trainy - trainpred)^2)), 
             sqrt(mean((data_testy - testpred)^2))),
    R2 = c(cor(data_trainy, trainpred)^2, 
           cor(data_testy, testpred)^2)
  )
)

# Save the SHAP analysis report
saveRDS(shap_report, "shap_analysis_report.rds")

# Print final summary
cat("\n========================================\n")
cat("SHAP ANALYSIS COMPLETE\n")
cat("========================================\n")
cat("Files generated:\n")
cat("1. shap_variable_importance.csv - Variable importance with percentages\n")
cat("2. shap_individual_contributions.csv - Individual observation contributions\n")
cat("3. climate_inequality_model.model - Trained XGBoost model\n")
cat("4. shap_analysis_report.rds - Complete SHAP analysis report\n")