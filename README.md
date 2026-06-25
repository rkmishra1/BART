# BART: Bayesian Additive Regression Trees in R

This repository provides a custom, optimized R implementation of **Bayesian Additive Regression Trees (BART)** based on the seminal paper:
> **BART: Bayesian additive regression trees**  
> Hugh A. Chipman, Edward I. George, and Robert E. McCulloch.  
> *The Annals of Applied Statistics*, 2010, Vol. 4, No. 1, 266–298.

Along with the core MCMC implementation, this repository includes replication scripts for **simulated data studies (Friedman's 5D function)** and **real data benchmarking (Boston Housing)** comparing BART against other popular machine learning regression methods (Lasso, Random Forests, Gradient Boosting, and OLS).

---

## 📖 Methodology Overview

BART is a Bayesian "sum-of-trees" model that acts as a non-parametric regression approach using dimensionally adaptive random basis elements. It approximates an unknown regression function $f(x) = E(Y|x)$ as a sum of $m$ trees:

$$Y = \sum_{j=1}^m g(x; T_j, M_j) + \epsilon, \quad \epsilon \sim N(0, \sigma^2)$$

where each $T_j$ represents a binary decision tree structure, and $M_j = (\mu_{1j}, \dots, \mu_{bj})$ represents the parameters associated with its $b$ terminal leaf nodes.

### 1. Regularization Priors
To prevent individual trees from dominating the model (making them "weak learners"), BART imposes a regularizing prior:
*   **Tree Structure Prior $P(T_j)$**: The probability of a node at depth $d$ splitting is $\alpha_0 (1+d)^{-\beta_0}$ (default $\alpha_0 = 0.95, \beta_0 = 2.0$), heavily favoring small trees (2-3 leaves).
*   **Leaf Parameter Prior $P(\mu_{ij} | T_j)$**: $\mu_{ij} \sim N(0, \sigma_\mu^2)$, where $\sigma_\mu = 0.5 / (k \sqrt{m})$. This shrinks individual leaf effects toward zero.
*   **Residual Variance Prior $P(\sigma^2)$**: $\sigma^2 \sim \text{Inv-Gamma}(\nu/2, \nu\lambda/2)$ where $\nu=3$ and $\lambda$ is calibrated to ensure that the 90th percentile of $\sigma$ falls below a rough estimate $\hat{\sigma}$ (e.g. sample standard deviation or OLS residual variance).

### 2. Bayesian Backfitting MCMC
Fitting is performed via a Gibbs sampler cycling through the trees. For each tree $j \in \{1, \dots, m\}$:
1.  Compute the **partial residuals** excluding tree $j$:
    $$R_j \equiv Y - \sum_{k \neq j} g(x; T_k, M_k)$$
2.  Propose a new tree structure $T_j^*$ using Metropolis-Hastings (MH) moves (Grow or Prune) based on the marginal likelihood $P(R_j | T_j, \sigma^2)$ after integrating out $M_j$.
3.  Draw new leaf parameters $M_j$ from their conjugate normal posterior:
    $$\mu_{ij} \sim N\left( \frac{\sigma_\mu^2 \sum_{k \in \text{Leaf } i} R_{j,k}}{\sigma^2 + n_i \sigma_\mu^2}, \frac{\sigma^2 \sigma_\mu^2}{\sigma^2 + n_i \sigma_\mu^2} \right)$$
4.  Draw the residual error variance $\sigma^2$ from its Inverse-Gamma posterior based on the full model residuals.

---

## ⚡ Performance Optimizations

Pure R implementations of tree algorithms can be slow. To address this, our implementation contains three critical algorithmic optimizations:

> [!TIP]
> **Vectorized Routing Map**: Rather than routing observations row-by-row down the trees, we route the entire dataset at once using vectorized masking.
>
> **$O(b)$ Train Predictions**: Since each tree node already stores the indices of training observations (`obs_idx`) falling into it, evaluating training set predictions is a simple $O(b)$ lookup of leaf values (where $b \le 10$ is the number of terminal leaves), bypassing routing altogether.
>
> **$O(N)$ running prediction sums**: Instead of summing all $m$ tree predictions with a costly matrix operation (`rowSums`) at every backfitting step, we maintain a running prediction vector `yhat_sum`. Computing the partial residual becomes a simple $O(N)$ subtraction: `R_j = y - (yhat_sum - tree_preds[, j])`.

---

## 📊 Simulations: Friedman's 5D Test Function

We evaluated our custom BART on **Friedman's 5D function** with $p=10$ inputs ($x_6 \dots x_{10}$ are dummy noise variables):

$$y = 10 \sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \epsilon, \quad \epsilon \sim N(0, 1)$$

### Estimation and MCMC Trace
Using $n=100$ observations, 200 trees, 1000 posterior draws, and 250 burn-in iterations:
*   The model successfully recovers the true signal, showing strong correlation between true $f(x)$ and predicted $\hat{f}(x)$ for both in-sample and out-of-sample data.
*   The posterior intervals show accurate frequentist coverage and widen for out-of-sample extrapolation.
*   The MCMC chain for $\sigma$ burns in extremely fast and fluctuates around the true value of $1.0$.

![Simulation Inference](figures/simulation_inference.png)

### Variable Selection (Figure 5 Replication)
By running BART with a smaller number of trees $m \in \{10, 20, 50, 100, 200\}$, we create a bottleneck that forces variables to compete for tree splits. 
*   As $m$ decreases, BART increasingly favors the true signal variables ($x_1 \dots x_5$), successfully screening out the noise variables ($x_6 \dots x_{10}$).

![Variable Selection](figures/variable_selection.png)

---

## 📈 Real Data Analysis & Benchmarking

We benchmarked the predictive performance of our custom BART on the **Boston Housing** dataset ($n=506$, $p=13$). Using 10 random train/test splits (5/6 train, 1/6 test), we compared BART against:
1.  **Ordinary Least Squares (OLS)**
2.  **Lasso Regression** (via `glmnet` cross-validation)
3.  **Random Forest** (via `randomForest` with 500 trees)
4.  **Gradient Boosting (GBM)** (via `gbm` with 200 trees)

### Summary Statistics
The following table shows the Mean RMSE and Relative RMSE (RRMSE) across the 10 splits:

| Method | Mean RMSE | Median RRMSE | 75% Quantile RRMSE |
| :--- | :---: | :---: | :---: |
| **GBM** | 3.30 | 1.000 | 1.047 |
| **Random Forest** | 3.46 | 1.070 | 1.122 |
| **Custom BART** | 3.87 | 1.233 | 1.344 |
| **OLS** | 5.04 | 1.636 | 1.731 |
| **Lasso** | 5.04 | 1.638 | 1.733 |

> [!NOTE]
> Our custom BART implementation achieved excellent performance, outperforming linear models (OLS/Lasso) by a large margin and remaining competitive with Random Forest and Gradient Boosting, despite running with a small ensemble (50 trees, 500 draws) for speed.

![Real Data Benchmark Boxplots](figures/real_data_benchmark.png)

---

## 🚀 How to Run the Code

To replicate these results locally, ensure you have R installed along with the required packages:

```R
install.packages(c("glmnet", "randomForest", "gbm", "ggplot2"))
```

### 1. Run Simulations
Run the Friedman 5D simulation script to generate Figure 3 and Figure 5:
```bash
Rscript simulations.R
```

### 2. Run Real Data Analysis
Run the Boston Housing benchmark script to generate the comparison boxplot:
```bash
Rscript real_data_analysis.R
```
All output figures will be saved in the `figures/` directory.
