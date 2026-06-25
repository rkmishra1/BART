# real_data_analysis.R - Real Data Analysis benchmarking on Boston Housing
# Compares Custom BART against Lasso, Random Forest, Gradient Boosting, and OLS.
# Generates RRMSE boxplots and tables equivalent to Figure 2 and Table 3.

source("bart.R")
library(glmnet)
library(randomForest)
library(gbm)

# Ensure output directory exists
dir.create("figures", showWarnings = FALSE)

# Set seed for reproducibility
set.seed(123)

# ==============================================================================
# 1. Load Boston Housing Dataset
# ==============================================================================
cat("Loading Boston Housing dataset...\n")
url <- "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
Boston <- tryCatch({
  read.csv(url)
}, error = function(e) {
  # Fallback to MASS package if offline
  if (requireNamespace("MASS", quietly = TRUE)) {
    data(Boston, package = "MASS")
    Boston
  } else {
    stop("Could not load Boston dataset from URL or MASS package.")
  }
})

# Separate features and target
y <- Boston$medv
X <- Boston[, -which(names(Boston) == "medv")]

# Ensure chas (categorical) is numeric indicator
if (is.factor(X$chas)) {
  X$chas <- as.numeric(X$chas) - 1
}

X <- as.matrix(X)

# ==============================================================================
# 2. Benchmark settings
# ==============================================================================
num_splits <- 10
methods <- c("OLS", "Lasso", "RandomForest", "GBM", "BART")
rmse_results <- matrix(NA, nrow = num_splits, ncol = length(methods))
colnames(rmse_results) <- methods

n <- nrow(X)
test_size <- round(n / 6) # 1/6 test set

cat("Starting benchmarking over", num_splits, "splits...\n")

for (split in 1:num_splits) {
  cat(paste("  Processing split", split, "...\n"))
  
  # Partition train/test
  test_idx <- sample(1:n, test_size)
  X_train <- X[-test_idx, ]
  y_train <- y[-test_idx]
  X_test <- X[test_idx, ]
  y_test <- y[test_idx]
  
  # --- 1. OLS ---
  lm_fit <- lm(y_train ~ ., data = data.frame(y_train, X_train))
  pred_ols <- predict(lm_fit, newdata = data.frame(X_test))
  rmse_results[split, "OLS"] <- sqrt(mean((y_test - pred_ols)^2))
  
  # --- 2. Lasso ---
  cv_lasso <- cv.glmnet(X_train, y_train)
  pred_lasso <- as.vector(predict(cv_lasso, newx = X_test, s = "lambda.min"))
  rmse_results[split, "Lasso"] <- sqrt(mean((y_test - pred_lasso)^2))
  
  # --- 3. Random Forest ---
  rf_fit <- randomForest(X_train, y_train, ntree = 500)
  pred_rf <- predict(rf_fit, X_test)
  rmse_results[split, "RandomForest"] <- sqrt(mean((y_test - pred_rf)^2))
  
  # --- 4. GBM ---
  gbm_fit <- gbm(y_train ~ ., data = data.frame(y_train, X_train), 
                 distribution = "gaussian", n.trees = 200, 
                 interaction.depth = 3, shrinkage = 0.1, verbose = FALSE)
  pred_gbm <- predict(gbm_fit, newdata = data.frame(X_test), n.trees = 200)
  rmse_results[split, "GBM"] <- sqrt(mean((y_test - pred_gbm)^2))
  
  # --- 5. Custom BART ---
  # We use fewer trees/iterations here to keep run time fast in pure R,
  # but enough to get good performance (50 trees, 500 posterior draws)
  bart_model <- bart_fit(X_train, y_train, X_test = X_test, 
                         num_trees = 50, ndpost = 500, nskip = 100)
  pred_bart <- bart_model$yhat_test_mean
  rmse_results[split, "BART"] <- sqrt(mean((y_test - pred_bart)^2))
}

# ==============================================================================
# 3. Analyze Results
# ==============================================================================
# Compute Relative RMSE (RRMSE)
# RRMSE = RMSE / min_method(RMSE) for each split
rrmse_results <- rmse_results / apply(rmse_results, 1, min)

# Compute Quantiles (50% and 75%)
rrmse_50 <- apply(rrmse_results, 2, median)
rrmse_75 <- apply(rrmse_results, 2, function(x) quantile(x, 0.75))

summary_table <- data.frame(
  Method = methods,
  Mean_RMSE = colMeans(rmse_results),
  Median_RRMSE = rrmse_50,
  Q75_RRMSE = rrmse_75
)

print(summary_table)

# Write summary table to a file for README inclusion
write.csv(summary_table, "figures/benchmark_summary.csv", row.names = FALSE)

# Generate RRMSE boxplots (matching Figure 2)
png("figures/real_data_benchmark.png", width = 800, height = 500, res = 120)
boxplot(rrmse_results, col = c("#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"),
        ylab = "Relative RMSE (RRMSE)", main = "Predictive Performance Comparison (Boston Housing)",
        las = 1, cex.axis = 0.8)
abline(h = 1.0, col = "red", lty = 2)
dev.off()

cat("Saved figures/real_data_benchmark.png\n")
cat("Real data analysis benchmarking completed successfully!\n")
