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
  int winner_y_hard[N_hard];  // number of sets won by the winner for match n
  int winner_y_clay[N_clay];  // number of sets won by the winner for match n
  int winner_y_grass[N_grass];  // number of sets won by the winner for match n
  int total_n_hard[N_hard];  // number of total sets played in match n
  int total_n_clay[N_clay];  // number of total sets played in match n
  int total_n_grass[N_grass];  // number of total sets played in match n
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
  // priors
  Lcorr ~ lkj_corr_cholesky(1);
  sigma ~ cauchy(0, 4); 
  beta ~ normal(0, 4);
  nu ~ exponential(0.1);
  alpha ~ multi_student_t(nu, Zero, Sigma);
  // sample
  winner_y_hard ~ binomial_logit(total_n_hard, X_skill_hard*(alpha_vector[1:J]) + (X1_hard - X2_hard)*beta);
  winner_y_clay ~ binomial_logit(total_n_clay, X_skill_clay*(append_row(alpha_vector[1:J], alpha_vector[(J+1):(2*J)])) + (X1_clay - X2_clay)*beta);
  winner_y_grass ~ binomial_logit(total_n_grass, X_skill_grass*(append_row(alpha_vector[1:J], alpha_vector[(2*J+1):(3*J)])) + (X1_grass - X2_grass)*beta);
}
generated quantities {
  // generate vector of length N_test of UNSEEN, HOLDOUT matches
  // yields number of expected sets won by player 1 drawing from the predictive posterior
  vector[N_test] p1_tilde;  // probability player 1 wins nth match in test set
  p1_tilde = inv_logit(X_skill_tilde*alpha_vector + (X1_tilde - X2_tilde)*beta);  // probability of set victory for player 1
  int y_tilde_1[N_test] = binomial_rng(N_tilde, p1_tilde);
  // sets won by player 2 are N_tilde (best of) minus # sets won by player 1
  //int y_tilde_2[N_test] = N_tilde - y_tilde_1;
}


