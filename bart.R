# bart.R - Custom Bayesian Additive Regression Trees (BART) in R
# Implementation of Chipman, George, and McCulloch (2010)

# Helper function to initialize a single-node tree
create_tree <- function(n_obs) {
  tree <- list()
  tree[["1"]] <- list(
    id = 1,
    is_terminal = TRUE,
    depth = 0,
    split_var = NA,
    split_val = NA,
    mu = 0,
    obs_idx = 1:n_obs
  )
  return(tree)
}

# Find all terminal node IDs
get_terminal_ids <- function(tree) {
  names(tree)[sapply(tree, function(node) node$is_terminal)]
}

# Find all prunable node IDs
# A node is prunable if it is non-terminal but both of its children are terminal.
get_prunable_ids <- function(tree) {
  ids <- names(tree)[!sapply(tree, function(node) node$is_terminal)]
  prunable <- c()
  for (id in ids) {
    id_num <- as.integer(id)
    left_child <- as.character(2 * id_num)
    right_child <- as.character(2 * id_num + 1)
    
    # Check if both children exist and are terminal
    if (!is.null(tree[[left_child]]) && tree[[left_child]]$is_terminal &&
        !is.null(tree[[right_child]]) && tree[[right_child]]$is_terminal) {
      prunable <- c(prunable, id)
    }
  }
  return(prunable)
}

# Helper to grow a tree by splitting a terminal node
grow_tree <- function(tree, target_id, split_var, split_val, X) {
  id_num <- as.integer(target_id)
  left_child_id <- as.character(2 * id_num)
  right_child_id <- as.character(2 * id_num + 1)
  
  node <- tree[[target_id]]
  obs <- node$obs_idx
  
  # Partition the observations
  left_obs <- obs[X[obs, split_var] <= split_val]
  right_obs <- obs[X[obs, split_var] > split_val]
  
  # Update target node to be internal
  tree[[target_id]]$is_terminal <- FALSE
  tree[[target_id]]$split_var <- split_var
  tree[[target_id]]$split_val <- split_val
  tree[[target_id]]$mu <- NA
  
  # Create children
  tree[[left_child_id]] <- list(
    id = 2 * id_num,
    is_terminal = TRUE,
    depth = node$depth + 1,
    split_var = NA,
    split_val = NA,
    mu = 0,
    obs_idx = left_obs
  )
  
  tree[[right_child_id]] <- list(
    id = 2 * id_num + 1,
    is_terminal = TRUE,
    depth = node$depth + 1,
    split_var = NA,
    split_val = NA,
    mu = 0,
    obs_idx = right_obs
  )
  
  return(tree)
}

# Helper to prune a tree by collapsing children back into the parent node
prune_tree <- function(tree, target_id) {
  id_num <- as.integer(target_id)
  left_child_id <- as.character(2 * id_num)
  right_child_id <- as.character(2 * id_num + 1)
  
  # Remove children from tree list
  tree[[left_child_id]] <- NULL
  tree[[right_child_id]] <- NULL
  
  # Make target node terminal again
  tree[[target_id]]$is_terminal <- TRUE
  tree[[target_id]]$split_var <- NA
  tree[[target_id]]$split_val <- NA
  tree[[target_id]]$mu <- 0
  
  return(tree)
}

# Fast vector-routed prediction for a single tree (used for test set)
predict_tree <- function(tree, X) {
  X <- as.matrix(X)
  n <- nrow(X)
  node_ids <- rep(1, n)
  
  is_term_map <- sapply(tree, function(node) node$is_terminal)
  terminal_flags <- is_term_map[as.character(node_ids)]
  
  while (any(!terminal_flags)) {
    non_term_idx <- which(!terminal_flags)
    unique_non_term_ids <- unique(node_ids[non_term_idx])
    
    for (id in unique_non_term_ids) {
      node <- tree[[as.character(id)]]
      rows_at_node <- which(node_ids == id)
      
      left_rows <- rows_at_node[X[rows_at_node, node$split_var] <= node$split_val]
      right_rows <- setdiff(rows_at_node, left_rows)
      
      node_ids[left_rows] <- 2 * id
      node_ids[right_rows] <- 2 * id + 1
    }
    
    terminal_flags <- is_term_map[as.character(node_ids)]
  }
  
  mu_map <- sapply(tree, function(node) node$mu)
  preds <- mu_map[as.character(node_ids)]
  return(preds)
}

# Extremely fast O(b) prediction for training data using pre-computed obs_idx
predict_tree_train <- function(tree, n_obs) {
  preds <- rep(0, n_obs)
  term_ids <- get_terminal_ids(tree)
  for (id in term_ids) {
    node <- tree[[id]]
    if (length(node$obs_idx) > 0) {
      preds[node$obs_idx] <- node$mu
    }
  }
  return(preds)
}

# Calculate the log marginal likelihood of a tree given partial residuals R and error variance sig2
# Omit terms that do not depend on tree structure
log_marginal_likelihood <- function(tree, R, sig2, sig2_mu) {
  term_ids <- get_terminal_ids(tree)
  log_lik <- 0
  
  for (id in term_ids) {
    node <- tree[[id]]
    n_eta <- length(node$obs_idx)
    if (n_eta == 0) next
    
    S_eta <- sum(R[node$obs_idx])
    
    term1 <- 0.5 * log(sig2 / (sig2 + n_eta * sig2_mu))
    term2 <- (sig2_mu * S_eta^2) / (2 * sig2 * (sig2 + n_eta * sig2_mu))
    log_lik <- log_lik + term1 + term2
  }
  
  return(log_lik)
}

# Main BART fitting function
bart_fit <- function(X, y, X_test = NULL, num_trees = 200, ndpost = 1000, nskip = 250,
                     k = 2, power = 2, base = 0.95, sigdf = 3, sigquant = 0.9) {
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  
  # 1. Scale y to range [-0.5, 0.5]
  y_min <- min(y)
  y_max <- max(y)
  y_range <- y_max - y_min
  y_scaled <- (y - y_min) / y_range - 0.5
  
  # 2. Calibrate prior parameters
  sig2_mu <- (0.5 / (k * sqrt(num_trees)))^2
  
  # Estimate starting sigma using linear regression if possible, otherwise sample sd
  fit_lm <- try(lm(y_scaled ~ X), silent = TRUE)
  if (!inherits(fit_lm, "try-error") && df.residual(fit_lm) > 0) {
    sighat <- summary(fit_lm)$sigma
  } else {
    sighat <- sd(y_scaled)
  }
  
  # Quantile-based calibration for Inv-Gamma prior of sig2
  C <- qchisq(1 - sigquant, df = sigdf)
  lambda <- C * sighat^2 / sigdf
  
  # Initialize parameter states
  sig2 <- sighat^2
  trees <- lapply(1:num_trees, function(i) create_tree(n))
  
  # Predictions matrix and running sums for each tree
  tree_preds <- matrix(0, nrow = n, ncol = num_trees)
  yhat_sum <- rep(0, n)
  
  # Test predictions support
  if (!is.null(X_test)) {
    X_test <- as.matrix(X_test)
    n_test <- nrow(X_test)
    test_tree_preds <- matrix(0, nrow = n_test, ncol = num_trees)
    test_yhat_sum <- rep(0, n_test)
    yhat_test_post <- matrix(0, nrow = n_test, ncol = ndpost)
  } else {
    yhat_test_post <- NULL
  }
  
  # Trace variables
  sig2_post <- rep(NA, ndpost)
  yhat_post <- matrix(0, nrow = n, ncol = ndpost)
  var_importance <- matrix(0, nrow = p, ncol = ndpost + nskip)
  
  # Total MCMC iterations
  total_iter <- ndpost + nskip
  
  for (iter in 1:total_iter) {
    # Loop through each tree to update its structure and leaf parameters
    for (j in 1:num_trees) {
      # Compute partial residuals R_j efficiently using running sum
      old_train_pred <- tree_preds[, j]
      R_j <- y_scaled - (yhat_sum - old_train_pred)
      
      # Metropolis-Hastings step for Tree Structure (T_j)
      tree <- trees[[j]]
      term_ids <- get_terminal_ids(tree)
      n_term <- length(term_ids)
      
      propose_grow <- if (n_term == 1) TRUE else (runif(1) < 0.5)
      
      proposed_tree <- tree
      
      if (propose_grow) {
        # Choose a terminal node to split
        target_id <- sample(term_ids, 1)
        node <- tree[[target_id]]
        
        # Select a split variable uniformly
        split_var <- sample(1:p, 1)
        vals <- X[node$obs_idx, split_var]
        uniq_vals <- sort(unique(vals))
        
        if (length(uniq_vals) >= 2) {
          split_val <- sample(uniq_vals[-length(uniq_vals)], 1)
          proposed_tree <- grow_tree(tree, target_id, split_var, split_val, X)
          
          depth <- node$depth
          p_split_current <- base * (1 + depth)^(-power)
          p_split_left_child <- base * (1 + depth + 1)^(-power)
          p_split_right_child <- base * (1 + depth + 1)^(-power)
          
          log_prior_ratio <- log(p_split_current) + 
                             2 * log(1 - p_split_left_child) - 
                             log(1 - p_split_current)
          
          p_prune_back <- 0.5
          p_grow_forward <- if (n_term == 1) 1.0 else 0.5
          
          n_prun_proposed <- length(get_prunable_ids(proposed_tree))
          n_splits <- length(uniq_vals) - 1
          
          log_proposal_ratio <- log(p_prune_back / n_prun_proposed) - 
                                log(p_grow_forward / (n_term * p * n_splits))
          
          log_lik_ratio <- log_marginal_likelihood(proposed_tree, R_j, sig2, sig2_mu) - 
                           log_marginal_likelihood(tree, R_j, sig2, sig2_mu)
          
          alpha <- exp(log_lik_ratio + log_prior_ratio + log_proposal_ratio)
          if (runif(1) < alpha) {
            tree <- proposed_tree
          }
        }
      } else {
        # Prune step
        prunable_ids <- get_prunable_ids(tree)
        if (length(prunable_ids) > 0) {
          target_id <- sample(prunable_ids, 1)
          node <- tree[[target_id]]
          
          proposed_tree <- prune_tree(tree, target_id)
          
          depth <- node$depth
          p_split_parent <- base * (1 + depth)^(-power)
          p_split_left <- base * (1 + depth + 1)^(-power)
          p_split_right <- base * (1 + depth + 1)^(-power)
          
          log_prior_ratio <- log(1 - p_split_parent) - 
                             (log(p_split_parent) + 2 * log(1 - p_split_left))
          
          p_grow_back <- if (length(get_terminal_ids(proposed_tree)) == 1) 1.0 else 0.5
          p_prune_forward <- 0.5
          
          original_split_var <- tree[[target_id]]$split_var
          uniq_vals <- sort(unique(X[node$obs_idx, original_split_var]))
          n_splits <- length(uniq_vals) - 1
          
          n_term_proposed <- length(get_terminal_ids(proposed_tree))
          n_prun_current <- length(prunable_ids)
          
          log_proposal_ratio <- log(p_grow_back / (n_term_proposed * p * n_splits)) - 
                                log(p_prune_forward / n_prun_current)
          
          log_lik_ratio <- log_marginal_likelihood(proposed_tree, R_j, sig2, sig2_mu) - 
                           log_marginal_likelihood(tree, R_j, sig2, sig2_mu)
          
          alpha <- exp(log_lik_ratio + log_prior_ratio + log_proposal_ratio)
          if (runif(1) < alpha) {
            tree <- proposed_tree
          }
        }
      }
      
      # Gibbs step for Leaf Parameters (M_j)
      term_ids <- get_terminal_ids(tree)
      for (id in term_ids) {
        node <- tree[[id]]
        obs <- node$obs_idx
        n_eta <- length(obs)
        
        if (n_eta > 0) {
          S_eta <- sum(R_j[obs])
          post_var <- 1 / (1 / sig2_mu + n_eta / sig2)
          post_mean <- post_var * (S_eta / sig2)
          tree[[id]]$mu <- rnorm(1, mean = post_mean, sd = sqrt(post_var))
        } else {
          tree[[id]]$mu <- rnorm(1, mean = 0, sd = sqrt(sig2_mu))
        }
      }
      
      trees[[j]] <- tree
      
      # Update training set prediction running sum
      new_train_pred <- predict_tree_train(tree, n)
      yhat_sum <- yhat_sum - old_train_pred + new_train_pred
      tree_preds[, j] <- new_train_pred
      
      # Update test set prediction running sum
      if (!is.null(X_test)) {
        old_test_pred <- test_tree_preds[, j]
        new_test_pred <- predict_tree(tree, X_test)
        test_yhat_sum <- test_yhat_sum - old_test_pred + new_test_pred
        test_tree_preds[, j] <- new_test_pred
      }
      
      # Track variable selection frequencies
      for (node in tree) {
        if (!node$is_terminal) {
          var_importance[node$split_var, iter] <- var_importance[node$split_var, iter] + 1
        }
      }
    }
    
    # Gibbs step for Error Variance (sig2) using yhat_sum
    residuals <- y_scaled - yhat_sum
    sum_sq_res <- sum(residuals^2)
    sig2 <- 1 / rgamma(1, shape = (sigdf + n) / 2, rate = (sigdf * lambda + sum_sq_res) / 2)
    
    if (iter > nskip) {
      post_idx <- iter - nskip
      sig2_post[post_idx] <- sig2 * y_range^2
      yhat_post[, post_idx] <- (yhat_sum + 0.5) * y_range + y_min
      
      if (!is.null(X_test)) {
        yhat_test_post[, post_idx] <- (test_yhat_sum + 0.5) * y_range + y_min
      }
    }
  }
  
  model <- list(
    trees_history = trees,
    sig2_post = sig2_post,
    yhat_post = yhat_post,
    yhat_mean = rowMeans(yhat_post),
    yhat_test_post = yhat_test_post,
    yhat_test_mean = if (!is.null(yhat_test_post)) rowMeans(yhat_test_post) else NULL,
    var_importance = var_importance,
    y_min = y_min,
    y_max = y_max,
    y_range = y_range,
    num_trees = num_trees,
    nskip = nskip,
    ndpost = ndpost,
    sig2_mu = sig2_mu
  )
  class(model) <- "bart"
  return(model)
}

# Predict function for new data
predict.bart <- function(object, X_new, ...) {
  X_new <- as.matrix(X_new)
  scaled_preds <- rep(0, nrow(X_new))
  for (tree in object$trees_history) {
    scaled_preds <- scaled_preds + predict_tree(tree, X_new)
  }
  preds_original <- (scaled_preds + 0.5) * object$y_range + object$y_min
  return(preds_original)
}
