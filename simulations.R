# simulations.R - Simulation study of BART on Friedman's 5D function
# Generates figures matching Figure 3 and Figure 5 in Chipman et al. (2010)

source("bart.R")

# Ensure output directory exists
dir.create("figures", showWarnings = FALSE)

# Set seed for reproducibility
set.seed(42)

# ==============================================================================
# 1. Friedman 5D Function definition
# ==============================================================================
friedman_f <- function(X) {
  10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5]
}

generate_friedman_data <- function(n, p = 10, sigma = 1.0) {
  X <- matrix(runif(n * p), nrow = n, ncol = p)
  f <- friedman_f(X)
  y <- f + rnorm(n, sd = sigma)
  return(list(X = X, y = y, f = f))
}

# ==============================================================================
# 2. Simulation 1: Point and Interval Estimation (Figure 3)
# ==============================================================================
cat("Running Simulation 1 (Estimation & MCMC trace)... \n")
n <- 100
p <- 10
train_data <- generate_friedman_data(n = n, p = p, sigma = 1.0)
test_data <- generate_friedman_data(n = n, p = p, sigma = 1.0)

# Fit custom BART model
# Using 200 trees, 1000 post-burn-in draws, 250 burn-in (skip)
model <- bart_fit(
  X = train_data$X, 
  y = train_data$y, 
  X_test = test_data$X, 
  num_trees = 200, 
  ndpost = 1000, 
  nskip = 250
)

# Compute posterior intervals (90%)
# In-sample
in_sample_mean <- model$yhat_mean
in_sample_lower <- apply(model$yhat_post, 1, quantile, probs = 0.05)
in_sample_upper <- apply(model$yhat_post, 1, quantile, probs = 0.95)

# Out-of-sample
out_sample_mean <- model$yhat_test_mean
out_sample_lower <- apply(model$yhat_test_post, 1, quantile, probs = 0.05)
out_sample_upper <- apply(model$yhat_test_post, 1, quantile, probs = 0.95)

# Save Figure 3 equivalent
png("figures/simulation_inference.png", width = 1200, height = 400, res = 120)
par(mfrow = c(1, 3))

# Plot 3a: In-sample
plot(train_data$f, in_sample_mean, xlab = "true f(x)", ylab = "in-sample fit",
     main = "(a) In-sample", pch = 19, col = "#1f77b4", ylim = range(c(in_sample_lower, in_sample_upper)))
abline(a = 0, b = 1, col = "red", lwd = 2)
segments(train_data$f, in_sample_lower, train_data$f, in_sample_upper, col = rgb(0,0,0,0.15))

# Plot 3b: Out-of-sample
plot(test_data$f, out_sample_mean, xlab = "true f(x)", ylab = "out-of-sample fit",
     main = "(b) Out-of-sample", pch = 19, col = "#2ca02c", ylim = range(c(out_sample_lower, out_sample_upper)))
abline(a = 0, b = 1, col = "red", lwd = 2)
segments(test_data$f, out_sample_lower, test_data$f, out_sample_upper, col = rgb(0,0,0,0.15))

# Plot 3c: MCMC Trace for Sigma
sigma_draws <- sqrt(model$sig2_post)
plot(sigma_draws, type = "l", col = "#7f7f7f", xlab = "mcmc iteration", ylab = "sigma",
     main = "(c) Sigma Trace")
abline(h = 1.0, col = "red", lwd = 2, lty = 2) # True sigma is 1.0
dev.off()

cat("Saved figures/simulation_inference.png\n")

# ==============================================================================
# 3. Simulation 2: Variable Selection (Figure 5)
# ==============================================================================
cat("Running Simulation 2 (Variable selection for m = 10, 20, 50, 100, 200)... \n")
n_var <- 500
var_data <- generate_friedman_data(n = n_var, p = p, sigma = 1.0)

tree_sizes <- c(10, 20, 50, 100, 200)
var_selection_results <- matrix(NA, nrow = length(tree_sizes), ncol = p)
rownames(var_selection_results) <- paste0("m=", tree_sizes)
colnames(var_selection_results) <- paste0("x", 1:p)

for (idx in seq_along(tree_sizes)) {
  m_size <- tree_sizes[idx]
  cat(paste("  Fitting BART with m =", m_size, "trees...\n"))
  
  m_model <- bart_fit(
    X = var_data$X, 
    y = var_data$y, 
    num_trees = m_size, 
    ndpost = 1000, 
    nskip = 250
  )
  
  # Calculate average split proportion for each variable (post burn-in)
  post_burn_imp <- m_model$var_importance[, (m_model$nskip + 1):(m_model$ndpost + m_model$nskip)]
  z_matrix <- apply(post_burn_imp, 2, function(col) {
    s <- sum(col)
    if (s > 0) col / s else rep(0, p)
  })
  
  var_selection_results[idx, ] <- rowMeans(z_matrix)
}

# Save Figure 5 equivalent
png("figures/variable_selection.png", width = 800, height = 500, res = 120)
plot(1:p, var_selection_results[1, ], type = "b", pch = 19, col = "red", ylim = c(0, max(var_selection_results) + 0.05),
     xaxt = "n", xlab = "Variable", ylab = "Percent Used", main = "Variable Importance by Tree Size")
axis(1, at = 1:p, labels = paste0("x", 1:p))
lines(1:p, var_selection_results[2, ], type = "b", pch = 19, col = "blue")
lines(1:p, var_selection_results[3, ], type = "b", pch = 19, col = "green3")
lines(1:p, var_selection_results[4, ], type = "b", pch = 19, col = "orange")
lines(1:p, var_selection_results[5, ], type = "b", pch = 19, col = "purple")

legend("topright", legend = paste0("m = ", tree_sizes), 
       col = c("red", "blue", "green3", "orange", "purple"), lty = 1, pch = 19, bty = "n")
dev.off()

cat("Saved figures/variable_selection.png\n")
cat("Simulation scripts completed successfully!\n")
