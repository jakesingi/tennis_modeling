library(dplyr)
library(lubridate)
library(tidyr)
library(stringr)
library(cmdstanr)
library(PlayerRatings)

# Matches from 2009 - 2017
# Number of matches: 25,609
# Number of players: 1,237

# DF of matches 
model_data = readRDS("./model_data.RDS")

# Player names
players = read.csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv") %>%
  mutate(player_id = as.character(player_id),
         full_name = paste(name_first, name_last))

# Gets match win probability for binomial sets model
compute_set_win_prob_given_g = function(game_win_prob) {
  set_win_prob = game_win_prob^6*(1 + 
                                    6*(1-game_win_prob) + 
                                    21*(1-game_win_prob)^2 + 
                                    56*(1-game_win_prob)^3 + 
                                    126*(1-game_win_prob)^4 + 
                                    252*game_win_prob*(1-game_win_prob)^5 + 
                                    504*(1-game_win_prob)^6*game_win_prob)
  return(set_win_prob)
}

# Computes match win probability given set win probability and best of (3 or 5)
compute_match_win_prob_given_s = function(set_win_prob, best_of) {
  if (best_of == 3) {
    match_win_prob = set_win_prob^2 + 2 * set_win_prob^2 * (1-set_win_prob) 
  } else {  # best of 5
    match_win_prob = set_win_prob^3*(1 + 3*(1-set_win_prob) + 6*(1-set_win_prob)^2)
  }
  return(match_win_prob)
}

# Trains (& optionally tests) model
# match_df: data frame of tennis matches
# model_file_path: the path to the STAN model
# test_month: the month in which we wish to make predictions
# winner_y: the "response" variable in the likelihood (often "games" or "sets")
# N: the "number of trials" variable in the likelihood (often "total_sets" or "games"), which each represenet the total sets and games in the match
# time_type: should always be "static", the "yearly" functionality has mostly been forgotten about
# model_variable: "games" or "sets" to inform at which level of granularity we start making predictions
# iters: a 2-vector of (num_iter_warmup, num_iter_sampling) for the MCMC algorithm
# testing: either T or F if we want to make predictions or not; testing = T ALSO gets glicko predictions
train_bayesian_bt = function(match_df, model_file_path, test_month, winner_y, N, time_type, model_variable, iters=c(1000, 1000), testing=F) {
  
  # Split into train/test
  train = match_df %>%
    filter(month < test_month)
  test = match_df %>%
    filter(month == test_month)
  
  # Clean everything up
  train = train %>%
    select(winner_id, loser_id, best_of, score, set_order, winner_y, N, winner_age, loser_age, N_months_since_last_match_winner,
           N_months_since_last_match_loser, surface, tourney_date, month, winner_rank, loser_rank)
  test = test %>%
    select(winner_id, loser_id, best_of, score, set_order, winner_y, N, winner_age, loser_age, N_months_since_last_match_winner,
           N_months_since_last_match_loser, surface, tourney_date, month, winner_rank, loser_rank)
  
  print(paste("Training period:", min(train$tourney_date), "-", max(train$tourney_date)))
  print(paste("Number of matches in training:", nrow(train)))
  print(paste("Num players:", length(unique(c(train$winner_id, train$loser_id)))))
  print(paste("Number of matches in testing:", nrow(test)))
  
  # Bind
  train_test = rbind(train, test)
  train_idx = (train_test$month < test_month) #& (train_test$winner_id %in% players_to_rate) & (train_test$loser_id %in% players_to_rate)
  test_idx = (train_test$month == test_month) #& (train_test$winner_id %in% players_to_rate) & (train_test$loser_id %in% players_to_rate)
  
  # Setup design matrices
  N_matches = nrow(train_test)  # number of matches
  J = length(unique(c(train_test$winner_id, train_test$loser_id)))  # number of players
  covariates = c("age", "age^2", "N_months_since_last_match")
  p = length(covariates)
  X_skill = matrix(0, nrow = N_matches, ncol = 3*J)  # Matrix for the player skills
  X1 = matrix(0, nrow = N_matches, ncol = p)  # Matrix for player 1 characteristics (age, age^2, and months since last match)
  X2 = matrix(0, nrow = N_matches, ncol = p)  # Matrix for player 2 characteristics (age, age^2, and months since last match)
  
  # Build X_skill, X1, and X2
  id_players = unique(c(train_test$winner_id, train_test$loser_id))
  colnames(X_skill) = c(paste0("overall_and_hard_skill_", id_players), paste0("clay_skill_", id_players), paste0("grass_skill_", id_players))
  colnames(X1) = covariates
  colnames(X2) = covariates
  
  for (i in 1:N_matches) {
    # Assign baseline (hard court) skills
    X_skill[i, paste0("overall_and_hard_skill_", train_test$winner_id[i]) ] = 1
    X_skill[i, paste0("overall_and_hard_skill_", train_test$loser_id[i])] = -1
    # If surface is clay, assign clay skills
    if (train_test[i, "surface"] == 2) {
      X_skill[i, paste0("clay_skill_", train_test$winner_id[i])]  = 1
      X_skill[i, paste0("clay_skill_", train_test$loser_id[i])]  = -1
      # If surface is grass, assign grass skills
    } else if (train_test[i, "surface"] == 3) {
      X_skill[i, paste0("grass_skill_", train_test$winner_id[i]) ]  = 1
      X_skill[i, paste0("grass_skill_", train_test$loser_id[i]) ]  = -1
    }
    # Build X1
    X1[i, 1] = train_test[i, "winner_age"]
    X1[i, 2] = train_test[i, "winner_age"]^2
    X1[i, 3] = train_test[i, "N_months_since_last_match_winner"]
    # Build X2
    X2[i, 1] = train_test[i, "loser_age"]
    X2[i, 2] = train_test[i, "loser_age"]^2
    X2[i, 3] = train_test[i, "N_months_since_last_match_loser"]
  }
  
  z1 = X_skill[, 1:J]
  z1[z1 == -1] = 0
  
  z2 = X_skill[, 1:J]
  z2[z2 == 1] = 0
  z2[z2 == -1] = 1
  
  # Separate into hard/clay/grass
  hard_idx = train_test$surface == 1
  clay_idx = train_test$surface == 2
  grass_idx = train_test$surface == 3
  
  # TRAIN design matrices
  X_skill_hard = X_skill[hard_idx & train_idx, 1:J]
  X_skill_clay = X_skill[clay_idx & train_idx, 1:(2*J)]
  X_skill_grass = X_skill[grass_idx & train_idx, c(1:J, (2*J+1):(3*J))]
  
  # TEST design matrices
  X_skill_tilde = X_skill[test_idx, ]
  X1_tilde = X1[test_idx, 3]
  X2_tilde = X2[test_idx, 3]
  
  # Prepare for TRAINING
  covariates = "N_months_since_last_match"
  p = length(covariates)
  
  # Collect remaining data
  winner_y_hard = train %>% filter(surface == 1) %>% pull(winner_y) 
  winner_y_clay = train %>% filter(surface == 2) %>% pull(winner_y) 
  winner_y_grass = train %>% filter(surface == 3) %>% pull(winner_y)  
  total_n_hard = train %>% filter(surface == 1) %>% pull(N) 
  total_n_clay = train %>% filter(surface == 2) %>% pull(N) 
  total_n_grass = train %>% filter(surface == 3) %>% pull(N) 
  
  winner_y = train %>% pull(winner_y)
  total_n = train %>% pull(N)
  total_n_tilde = test %>% pull(N)
  
  Nhard = nrow(X_skill_hard)  # number of matches on hard
  Nclay = nrow(X_skill_clay)
  Ngrass = nrow(X_skill_grass)
  if (model_variable == "sets") {
    N_tilde = test$best_of
  } else {
    N_tilde = 13 * test$best_of
  }
  
  # Number of test matches
  N_test = nrow(X_skill_tilde)
  
  # Fit model
  print("ESTIMATING PARAMETERS")
  fit = cmdstan_model(model_file_path, cpp_options = list(stan_threads = TRUE))
  model_data = list(J = J, N_train = nrow(train), N_hard = Nhard, N_clay = Nclay, N_grass = Ngrass, 
                    X_skill_hard = X_skill_hard, X_skill_clay = X_skill_clay, X_skill_grass = X_skill_grass, 
                    X1_hard = X1[hard_idx & train_idx, 3], X2_hard = X2[hard_idx & train_idx, 3], 
                    X1_clay = X1[clay_idx & train_idx, 3], X2_clay = X2[clay_idx & train_idx, 3], 
                    X1_grass = X1[grass_idx & train_idx, 3], X2_grass = X2[grass_idx & train_idx, 3],
                    winner_y_hard = winner_y_hard, winner_y_clay = winner_y_clay, winner_y_grass = winner_y_grass,
                    total_n_hard = total_n_hard, total_n_clay = total_n_clay, total_n_grass = total_n_grass, 
                    N_test = N_test, N_tilde = N_tilde, X_skill_tilde = X_skill_tilde, X1_tilde = X1_tilde, X2_tilde = X2_tilde,
                    z1 = z1[train_idx, ], z2 = z2[train_idx, ], X_skill = X_skill[train_idx, ], X1 = X1[train_idx, 3], X2 = X2[train_idx, 3],
                    total_n = total_n, winner_y = winner_y, total_n_tilde = total_n_tilde, 
                    z1_tilde = z1[test_idx, ], z2_tilde = z2[test_idx, ])
  fit = fit$sample(model_data, chains = 4, parallel_chains = 4, threads_per_chain = 2, refresh = 200, iter_warmup=iters[1], iter_sampling=iters[2], init = function() list(alpha = runif(3*J, -1, 1), nu = runif(1, 0, 50))) 
  train_time = fit$time()$total
  # fit = fit$variational(model_data)  # VB
  skill_draws = fit$draws()
  
  # chain diagnostics
  chain_diagnostics = fit$diagnostic_summary()
  
  # Save diagnostics in case we want to check
  skill_summaries = summarize_draws(bind_draws(skill_draws))
  
  # Collect draws
  if (time_type == "static") {
    idx_skills = grepl("alpha\\[.*\\]", skill_summaries$variable)
    skills = skill_draws[, , idx_skills]
    out2 = matrix(nrow = dim(skills)[1] * dim(skills)[2], ncol = dim(skills)[3])
    # Extract draws
    for (i in 1:(3*J)) {
      out2[, i] = as.vector(rbind(skills[, , i]))
    }
    colnames(out2) = colnames(X_skill)
    # Get posterior means of the skills
    alpha_hard = colMeans(out2[, 1:J])
    alpha_clay = colMeans(out2[, (J+1):(2*J)])
    alpha_grass = colMeans(out2[, (2*J+1):(3*J)])
    
    # Organize posterior means with player IDs
    overall_skills = data.frame(out2)
    overall_skills["draw"] = 1:(dim(out2)[1])
    overall_skills = pivot_longer(overall_skills, cols = -c("draw"))
    overall_skills = overall_skills %>%
      mutate(skill_type = str_extract(name, "(hard|clay|grass)"))
    
    # Add column for player id
    overall_skills = overall_skills %>%
      mutate(player_id = str_extract(name, "[0-9]+"))
    
    # Add clay and grass skills to baseline 
    # Pivot for plotting
    # Get player names
    overall_skills_by_surface = overall_skills %>%
      group_by(player_id, draw) %>%
      mutate(clay_skill = value[skill_type == "clay"] + value[skill_type == "hard"],
             grass_skill = value[skill_type == "grass"] + value[skill_type == "hard"],
             hard_skill = value[skill_type == "hard"]) %>%  ## NEWLY ADDED like in year over year testing. confirm it works
      #rename(hard_skill = value) %>%
      select(-skill_type, -name, -value) %>%
      #distinct(player_id, .keep_all = T) %>%
      pivot_longer(., cols = c("hard_skill", "clay_skill", "grass_skill"), names_to = "skill_type") %>%
      left_join(players, by = "player_id")
    
    posterior_mean_df = overall_skills_by_surface %>%
      group_by(player_id, full_name, skill_type) %>%
      summarize(posterior_mean = mean(value),
                posterior_sd = sd(value),
                lower.025_q = quantile(value, .025),
                upper.975_q = quantile(value, .975),
                posterior_median = median(value))
  } else if (time_type == "yearly") {
    idx_hard_skills = grepl("alpha\\[(\\d{2,}|[2-9]),.*,1\\]", skill_summaries$variable)  # Get all skills, EXCLUDING initial alpha
    idx_clay_skills = grepl(paste0("alpha\\[(\\d{2,}|[2-9]),.*,2\\]"), skill_summaries$variable)
    idx_grass_skills = grepl(paste0("alpha\\[(\\d{2,}|[2-9]),.*,3\\]"), skill_summaries$variable)
    hard_skills = skill_draws[, , idx_hard_skills]
    clay_skills = skill_draws[, , idx_clay_skills]
    grass_skills = skill_draws[, , idx_grass_skills]
    hard_out = matrix(nrow = dim(hard_skills)[1] * dim(hard_skills)[2], ncol = dim(hard_skills)[3])
    clay_out = matrix(nrow = dim(clay_skills)[1] * dim(clay_skills)[2], ncol = dim(clay_skills)[3])
    grass_out = matrix(nrow = dim(grass_skills)[1] * dim(grass_skills)[2], ncol = dim(grass_skills)[3])
    # Extract draws
    for (i in 1:ncol(hard_out)) {
      hard_out[, i] = as.vector(rbind(hard_skills[, , i]))
      clay_out[, i] = as.vector(rbind(clay_skills[, , i]))
      grass_out[, i] = as.vector(rbind(grass_skills[, , i]))
    }
    colnames(hard_out) = paste(rep(colnames(X_skill_hard), each=length(years_train)), years_train, sep="_")
    colnames(clay_out) = paste(rep(colnames(X_skill_clay)[(J+1):(2*J)], each=length(years_train)), years_train, sep="_")
    colnames(grass_out) = paste(rep(colnames(X_skill_grass)[(J+1):(2*J)], each=length(years_train)), years_train, sep="_")
    hard_out = as.data.frame(hard_out)
    clay_out = as.data.frame(clay_out)
    grass_out = as.data.frame(grass_out)
    
    overall_skills = cbind(hard_out, clay_out, grass_out)
    overall_skills['draw'] = 1:nrow(overall_skills)
    overall_skills = pivot_longer(overall_skills,cols = -c("draw"))
    overall_skills = overall_skills %>%
      mutate(skill_type = str_extract(name, "(hard|clay|grass)"),
             year = str_extract(name, "[0-9]{4}$"),
             player_id = str_extract(name, "[0-9]{6}"))
    
    overall_skills_by_surface = overall_skills %>%
      group_by(player_id, year, draw) %>%
      mutate(clay_skill = value[skill_type == "clay"] + value[skill_type == "hard"],
             grass_skill = value[skill_type == "grass"] + value[skill_type == "hard"],
             hard_skill = value[skill_type == "hard"]) %>%
      select(-skill_type, -name, -value) %>%
      #distinct(player_id, .keep_all = T) %>%
      pivot_longer(., cols = c("hard_skill", "clay_skill", "grass_skill"), names_to = "skill_type") %>%
      left_join(players, by = "player_id")
    
    posterior_mean_df = overall_skills_by_surface %>%
      group_by(player_id, full_name, year, skill_type) %>%
      summarize(posterior_mean = mean(value),
                posterior_sd = sd(value),
                lower.025_q = quantile(value, .025),
                upper.975_q = quantile(value, .975),
                posterior_median = median(value)) %>%
      mutate(year = as.numeric(year))
  }
  
  if (testing) {
    # Get GLICKO-2 ratings
    glicko_train = train %>%
      mutate(y = 1, tourney_date = as.numeric(tourney_date)) %>%
      select(tourney_date, winner_id, loser_id, y)
    
    glicko_out = glicko2(glicko_train)  # defaults
    
    glicko_ratings = glicko_out[[1]] %>%
      mutate(Player = as.character(Player)) %>%
      inner_join(players %>% select(player_id, full_name), by = c("Player" = "player_id"))
    
    # Make glicko predictions on the test set
    glicko_test = test %>%
      mutate(tourney_date = as.numeric(tourney_date)) %>%
      select(tourney_date, winner_id, loser_id)
    
    # Get Glicko-2 predictions
    # If player has played less than tng matches (15 by default), set their rating to 2200, deviation of 300 (default) and predict. 
    # gamma=0 means no player 1 advantage
    glicko_predict = predict(glicko_out, glicko_test, tng=15,trat=c(2200, 300), gamma=0)
    glicko_test$p1_win_prob = glicko_predict
    glicko_acc = mean(glicko_predict > 0.5)
    
    # Collect p1 win probabilities
    probs_idx_1 = grepl("p1_tilde", skill_summaries$variable)
    probs1_draws = skill_draws[, , probs_idx_1]
    probs1 = matrix(nrow=dim(probs1_draws)[1] * dim(probs1_draws)[2], ncol = dim(probs1_draws)[3])
    for (i in 1:(sum(probs_idx_1==T))) {
      probs1[, i] = rbind(probs1_draws[, , i])
    }
    
    if (model_variable == "sets") {
      # Add SET win prob to test frame
      p1_set_win_prob_post_means = colMeans(probs1)
      test$pp_p1_set_win_prob = p1_set_win_prob_post_means
      # Add MATCH win prob to test frame
      best_of_vec = rep(test$best_of, each=nrow(probs1))
      s_vec = as.vector(probs1)
      match_win_probs = mapply(compute_match_win_prob_given_s, s_vec, best_of_vec)
      match_win_prob_mat = matrix(match_win_probs, nrow=nrow(probs1), ncol = nrow(test), byrow = F)
      match_win_prob_post_means = colMeans(match_win_prob_mat)
      test$pp_p1_match_win_prob = match_win_prob_post_means
    } else if (model_variable == "games") {
      # Have to get the corresponding set win probabilities given the game win probabilities FIRST
      p1_set_win_prob_post_means = colMeans(compute_set_win_prob_given_g(probs1))
      test$pp_p1_set_win_prob = p1_set_win_prob_post_means
      # Add MATCH win prob to test frame
      g_vec = as.vector(probs1)
      s_vec = compute_set_win_prob_given_g(g_vec)
      best_of_vec = rep(test$best_of, each=nrow(probs1))
      match_win_probs = mapply(compute_match_win_prob_given_s, s_vec, best_of_vec)
      match_win_prob_mat = matrix(match_win_probs, nrow=nrow(probs1), ncol = nrow(test), byrow = F)
      match_win_prob_post_means = colMeans(match_win_prob_mat)
      test$pp_p1_match_win_prob = match_win_prob_post_means
    }
    
    # Get model & ranking test accuracies
    model_acc = mean(test$pp_p1_match_win_prob > 0.5) 
    ranking_acc = mean(ifelse(is.na(test$winner_rank), 0, ifelse(is.na(test$loser_rank), 1, ifelse(test$winner_rank < test$loser_rank, 1, 0))))
    
    res = list(skill_draws=skill_draws, skill_summaries=skill_summaries, pp_p1_win_prob_draws = probs1, 
               posterior_mean_df=posterior_mean_df, overall_skills_by_surface=overall_skills_by_surface, test=test, glicko_test=glicko_test,
               model_acc = model_acc, ranking_acc = ranking_acc, glicko_acc = glicko_acc,
               train_time = train_time, chain_diagnostics = chain_diagnostics,
               X_skill = X_skill)
  } else {
    res = list(skill_draws=skill_draws, skill_summaries=skill_summaries, 
               posterior_mean_df=posterior_mean_df, overall_skills_by_surface=overall_skills_by_surface, 
               train_time = train_time, chain_diagnostics=chain_diagnostics,
               X_skill = X_skill)
  }
  return(res)
}

# Fit the NegBin stochastic variance model with the player erraticities
# Train on 2009 and test on January 2010
experiment1 = train_bayesian_bt(match_df = model_data,
                               model_file_path = "./multivariate_glicko.stan",
                               test_month = "2009-03-01", 
                               winner_y = "sets_won_loser",
                               N = "total_sets",
                               time_type = "static",
                               model_variable = "sets",
                               iters = c(100, 100),
                               testing = T)

# Fit the Binomial model with the player erraticities
# Train on 2009 and test on January 2010
experiment2 = train_bayesian_bt(match_df = model_data,
                               model_file_path = "./correlated_skills_multi_t.stan",
                               test_month = "2009-03-01", 
                               winner_y = "w_games",
                               N = "games",
                               time_type = "static",
                               model_variable = "games",
                               iters = c(100, 100),
                               testing = T)
