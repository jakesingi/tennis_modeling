library(cmdstanr)

# Number of matches: 114
# Number of players: 15

# List with inputs to the model
model_data = readRDS("model_data.RDS")

# Fit the model
fit = cmdstan_model("./multivariate_glicko.stan", cpp_options = list(stan_threads = TRUE))
fit = fit$sample(model_data, chains = 2, parallel_chains = 2, threads_per_chain = 4, refresh = 200, iter_warmup = 250, iter_sampling=250)