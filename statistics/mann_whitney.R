# Load necessary library


# Read CSV
set.seed(123)  # for reproducibility

# Read CSV
df <- read.csv("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/only_surprisals_partial.csv", stringsAsFactors = FALSE)

# Get all unique model names
model_names <- unique(df$Model_Name)
combinations <- combn(model_names, 2, simplify = FALSE)

# Function to compute CI via bootstrapping
bootstrap_ci <- function(x, y, n_iter = 1000, conf = 0.95) {
  diffs <- replicate(n_iter, {
    x_samp <- sample(x, replace = TRUE)
    y_samp <- sample(y, replace = TRUE)
    median(x_samp) - median(y_samp)
  })
  alpha <- (1 - conf) / 2
  quantile(diffs, c(alpha, 1 - alpha))
}

# Store results
results <- data.frame(
  Group1 = character(),
  Group2 = character(),
  U = numeric(),
  Z = numeric(),
  r = numeric(),
  p_raw = numeric(),
  CI_lower = numeric(),
  CI_upper = numeric(),
  stringsAsFactors = FALSE
)

# Run tests
for (pair in combinations) {
  g1 <- pair[1]
  g2 <- pair[2]
  x <- df$Surprisals[df$Model_Name == g1]
  y <- df$Surprisals[df$Model_Name == g2]
  
  test <- wilcox.test(x, y, exact = FALSE)
  U <- as.numeric(test$statistic)
  
  n1 <- length(x)
  n2 <- length(y)
  mu_U <- n1 * n2 / 2
  sigma_U <- sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
  Z <- (U - mu_U) / sigma_U
  r <- Z / sqrt(n1 + n2)
  
  # Bootstrap CI
  ci_vals <- bootstrap_ci(x, y, n_iter = 1000)
  
  results <- rbind(results, data.frame(
    Group1 = g1,
    Group2 = g2,
    U = U,
    Z = Z,
    r = r,
    p_raw = test$p.value,
    CI_lower = ci_vals[1],
    CI_upper = ci_vals[2],
    stringsAsFactors = FALSE
  ))
}

# Holm correction
results$p_holm <- p.adjust(results$p_raw, method = "holm")

# View results
print(results)
write.csv(results, "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/analysis/mann_whitney_results.csv", row.names = FALSE)


