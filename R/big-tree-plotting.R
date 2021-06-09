require(tidyverse)

setwd("~/Documents/Salo/BigTrees")

height_40m_obj_max <- read_csv("tcsi_height_03m_40m_obj_max.csv")

height_40m_obj_n <- read_csv("tcsi_height_03m_40m_obj_n.csv")


ggplot(data = height_40m_obj, aes(x = cfo, y = lidar)) +
  geom_hex()




rdpareto <- function(n, lambda, max) {
  x <- 9:max
  sample(x, size = n, replace = T, prob = ddpareto(x, lambda))
}

# Density function for the discrete Pareto distribution with parameter lambda

ddpareto <- function(x, lambda) {
  require(VGAM)
  x^-lambda / zeta(lambda)
}

# A function to return negative 2 times the log-likelihood of data under a
# discrete Pareto distribution with scaling parameter lambda.

dplik <- function(data, lambda) {
  sum(log(ddpareto(x = data, lambda = lambda)))
}


calc.lambda <- function(gap_data, se = TRUE, nbootstrap = 1000) {
  # lambda
  lambda.est <- optimize(dplik, data = gap_data, lower = 1.0001, upper = 20, maximum = T, tol = 0.000001)$maximum
  if (se == TRUE) {
    # bootstrap se of lambda
    ngaps <- length(gap_data)
    boot <- matrix(NA, nrow = nbootstrap, ncol = 1)
    for (i in 1:nbootstrap) {
      x <- sample(gap_data, size = ngaps, replace = T)
      boot[i] <- optimize(dplik, data = x, lower = 1.0001, upper = 20, maximum = T)$maximum
    }
    se_hi <- as.vector(quantile(boot, 0.975))
    se_low <- as.vector(quantile(boot, 0.025))
    return(c(lambda.est, se_hi, se_low)) # return max loglik estimate of lambda and the bootstrap std errs
  } else {
    (return(lambda.est))
  }
}


tcsi_clump_lambda <- calc.lambda(height_40m_obj_n$cfo, se = FALSE, nbootstrap = 1000)

tsci_clump_hist <- hist(height_40m_obj_n$cfo, br = seq(0, max(height_40m_obj_n$cfo), 1), plot = F)

plot(y = tsci_clump_hist$counts, x = tsci_clump_hist$breaks[2:length(tsci_clump_hist$breaks)], log = "xy", axes = F, col = "red")

tcsi_lambda_df <- data.frame(tsci_clump_hist$counts, tsci_clump_hist$breaks[2:length(tsci_clump_hist$breaks)] * 9)
names(tcsi_lambda_df) <- c("counts", "breaks_m2")

ggplot(data = tcsi_lambda_df) +
  geom_point(aes(y = counts, x = breaks_m2), alpha = 0.3, color = "blue", stroke = 0, shape = 16, size = 3) +
  # scale_x_continuous(trans='log10', expand=c(0,0)) +
  # scale_y_continuous(trans='log10', expand=c(0,0)) +
  geom_abline(intercept = 2e5, slope = dplik(breaks, tcsi_clump_lambda)) +
  ylab("Clump Frequency") +
  xlab("Clump size (m2)") +
  theme_bw()

simdata1 <- rdpareto(1e6, 1.5, 7000)
sim1 <- hist(simdata1, br = seq(9, max(simdata1), 1), plot = F)
lambda_sim_df1 <- data.frame(counts = sim1$counts, breaks_m2 = sim1$breaks[2:length(sim1$breaks)])

simdata2 <- rdpareto(1e6, 2.5, 7000)
sim2 <- hist(simdata2, br = seq(9, max(simdata2), 1), plot = F)
lambda_sim_df2 <- data.frame(counts = sim2$counts, breaks_m2 = sim2$breaks[2:length(sim2$breaks)])

ggplot(data = lambda_sim_df1 %>% mutate(group = "lambda=1.5") %>%
  bind_rows(lambda_sim_df2 %>% mutate(group = "lambda=2.5")) %>%
  bind_rows(tcsi_lambda_df %>% mutate(group = "tcsi"))) +
  geom_point(aes(y = counts, x = breaks_m2, color = group), alpha = 0.3, stroke = 0, shape = 16, size = 3) +
  scale_x_log10() +
  scale_y_continuous(trans = "log10", expand = c(0, 0)) +
  scale_color_manual(values = c("blue", "red", "black")) +
  # geom_abline(intercept = 2e5,slope = dplik(breaks, tcsi_clump_lambda)) +
  ylab("Clump Frequency") +
  xlab("Clump size (m2)") +
  theme_bw()
