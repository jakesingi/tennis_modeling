functions {
  // normalizing function
  row_vector normalize(row_vector x) {
    row_vector[num_elements(x)] out;
    //for (i in 1:num_elements(x)) {
    //  out[i] = x[i] / sum(x);
    //}
    out = x / sum(x);
    return out;
  }
  // Compute win probability of set given win probability of game
  matrix get_set_win_prob(vector g, int N) {
    matrix[N, 14] out;
    matrix[N, 14] out_normalized;
    vector[N] p60 = g^6;  // win 6-0
    vector[N] p61 = choose(6, 5) .* g^5 .* (1-g) .* g;  // win 6-1
    vector[N] p62 = choose(7, 5) .* g^5 .* (1-g)^2 .* g;  // win 6-2
    vector[N] p63 = choose(8, 5) .* g^5 .* (1-g)^3 .* g;  // win 6-3
    vector[N] p64 = choose(9, 5) .* g^5 .* (1-g)^4 .* g; // win 6-4
    vector[N] p75 = choose(10, 5) .* g^5 .* (1-g)^5 .* g^2; // win 7-5
    vector[N] p76 = choose(10, 5) .* g^5 .* (1-g)^5 .* choose(2, 1) .* g .* (1-g) .* g; // win 7-6
    vector[N] p06 = (1-g)^6;  // lose 0-6
    vector[N] p16 = choose(6, 1) .* (1-g)^5 .* g .* (1-g);  // lose 1-6
    vector[N] p26 = choose(7, 5) .* (1-g)^5 .* g^2 .* (1-g);  // lose 2-6
    vector[N] p36 = choose(8, 5) .* (1-g)^5 .* g^3 .* (1-g);  // lose 3-6
    vector[N] p46 = choose(9, 5) .* (1-g)^5 .* g^4 .* (1-g);  // lose 4-6
    vector[N] p57 = choose(10, 5) .* (1-g)^5 .* g^5 .* (1-g)^2;  // lose 5-7
    vector[N] p67 = choose(10, 5) .* (1-g)^5 .* g^5 .* choose(2, 1) .* (1-g) .* g .* (1-g);  // lose 6-7
    out = append_col(append_col(append_col(append_col(append_col(append_col(append_col(append_col(append_col(append_col(append_col(append_col(append_col(p60, p61), p62), p63), p64), p75), p76), p06), p16), p26), p36), p46), p57), p67);
    out_normalized = out;
    for (i in 1:N) {
      out_normalized[i] = normalize(out[i]);
    }
    return out_normalized;
  }
}

data {
  int<lower=0> J;         // total number of players
  int<lower=0> N_hard;         // number of matches
  int<lower=0> N_clay;
  int<lower=0> N_grass;
  int<lower=0> N_test;  // number of matches in the test set
  int N_tilde[N_test];  // best of sets in n-tilde-th match
  matrix[N_test, 3*J] X_skill_tilde;  // design matrix for the test set
  vector[N_test] X1_tilde;  // vector of time off for p1 for the test set
  vector[N_test] X2_tilde;  // vector of time off for p2 for the test set
  matrix[N_hard, J] X_skill_hard;  // design matrix for the skills on hard
  matrix[N_clay, 2*J] X_skill_clay;  // design matrix for the skills on clay
  matrix[N_grass, 2*J] X_skill_grass;  // design matrix for the skills on grass
  vector[N_hard] X1_hard;
  vector[N_hard] X2_hard;
  vector[N_clay] X1_clay;
  vector[N_clay] X2_clay;
  vector[N_grass] X1_grass;
  vector[N_grass] X2_grass;
  int<lower=1, upper=14> winner_y_hard[N_hard];  // number of sets won by the winner for match n
  int<lower=1, upper=14> winner_y_clay[N_clay];  // number of sets won by the winner for match n
  int<lower=1, upper=14> winner_y_grass[N_grass];  // number of sets won by the winner for match n
  int total_n_hard[N_hard];  // number of total sets played in match n
  int total_n_clay[N_clay];  // number of total sets played in match n
  int total_n_grass[N_grass];  // number of total sets played in match n
  int<lower=1> grainsize;
}
transformed data {
  vector[3] Zero = rep_vector(0, 3);
}
parameters {
  cholesky_factor_corr[3] Lcorr;
  vector<lower=0>[3] sigma;
  row_vector[3] alpha[J];  // player skills
  real beta;  // coefficent on N_months_since_last_match
  real<lower=0> nu;  // degrees of freedom
}
transformed parameters {
  // define correlation and covariance matrices
  corr_matrix[3] R;
  cov_matrix[3] Sigma;
  R = multiply_lower_tri_self_transpose(Lcorr);  // does Lcorr %*% t(Lcorr)
  Sigma = quad_form_diag(R, sigma);  // does sigmaI %*% R %*% sigmaI
  vector[3*J] alpha_vector = to_vector(to_matrix(alpha));
}
model {
  vector[N_hard] p1_game_win_hard;
  vector[N_clay] p1_game_win_clay;
  vector[N_grass] p1_game_win_grass;
  matrix[N_hard, 14] p1_set_win_hard;
  matrix[N_clay, 14] p1_set_win_clay;
  matrix[N_grass, 14] p1_set_win_grass;
  // priors
  Lcorr ~ lkj_corr_cholesky(1);
  sigma ~ cauchy(0, 4); 
  beta ~ normal(0, 4);
  nu ~ exponential(0.1);
  alpha ~ multi_student_t(nu, Zero, Sigma);
  p1_game_win_hard = inv_logit(X_skill_hard*(alpha_vector[1:J]) + (X1_hard - X2_hard)*beta);
  p1_game_win_clay = inv_logit(X_skill_clay*(append_row(alpha_vector[1:J], alpha_vector[(J+1):(2*J)])) + (X1_clay - X2_clay)*beta);
  p1_game_win_grass = inv_logit(X_skill_grass*(append_row(alpha_vector[1:J], alpha_vector[(2*J+1):(3*J)])) + (X1_grass - X2_grass)*beta);
  p1_set_win_hard = get_set_win_prob(p1_game_win_hard, N_hard);
  p1_set_win_clay = get_set_win_prob(p1_game_win_clay, N_clay);
  p1_set_win_grass = get_set_win_prob(p1_game_win_grass, N_grass);
  // sample
  for (i in 1:N_hard) {
    winner_y_hard[i] ~ categorical(to_vector(p1_set_win_hard[i]));
  }
  for (j in 1:N_clay) {
    winner_y_clay[j] ~ categorical(to_vector(p1_set_win_clay[j]));
  }
  for(k in 1:N_grass) {
    winner_y_grass[k] ~ categorical(to_vector(p1_set_win_grass[k]));
  }
}
generated quantities {
  // generate vector of length N_test of UNSEEN, HOLDOUT matches
  // yields number of expected sets won by player 1 drawing from the predictive posterior
  vector[N_test] p1_tilde;  // probability player 1 wins nth match in test set
  p1_tilde = inv_logit(X_skill_tilde*alpha_vector + (X1_tilde - X2_tilde)*beta);  // probability of GAME victory for player 1
  //int y_tilde_1[N_test] = bernoulli_logit_rng(X_skill_tilde*alpha_vector + (X1_tilde - X2_tilde)*beta);
  // sets won by player 2 are N_tilde (best of) minus # sets won by player 1
  //int y_tilde_2[N_test] = N_tilde - y_tilde_1;
}


