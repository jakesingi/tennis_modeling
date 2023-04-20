functions {
  // Computes the log-PDF of the negative binomial distribution with y failures until r successes with success probability p
  // vetorized
  real negbin_vec_lpmf(int[] y, vector p, int[] r) {
    int n = num_elements(y);
    vector[n] y_ = to_vector(y);
    vector[n] r_ = to_vector(r);
    return sum(lchoose(y_ + r_ - 1, y_) + r_.*log(p) + y_.*log1m(p));
  }
  // element-by-element
  real negbin_lpmf(int y, real p, int r) {
    real lp = lchoose(y + r - 1, y) + r*log(p) + y*log1m(p);
    return lp;
  }
}
data {
  int<lower=0> J;         // number of players
  int<lower=0> N_train;  // number of matches
  matrix[N_train, J] z1;  // indicator matrix where z[i, j] = 1 if player j is player 1 in match i
  matrix[N_train, J] z2;  // indicator matrix where z[i, j] = 1 if player j is player 2 in match i
  matrix[N_train, 3*J] X_skill;  // design matrix for player skills
  vector[N_train] X1;  // months passed since player 1 last played before match i 
  vector[N_train] X2;  // months passed since player 2 last played before match i 
  int total_n[N_train];  // whether the match was best of 3 or 5 sets
  int winner_y[N_train];  // number of sets LOST by the winner of the match
}
transformed data {
  vector[3] Zero = rep_vector(0, 3);
  // Build up requisite number of set victories, which is 2 for a best of 3 match or 3 for a best of 5 match
  int<lower=2, upper=3> r[N_train];
  for (i in 1:N_train) {
    if (total_n[i] == 3) {
      r[i] = 2;
    } else {
      r[i] = 3;
    }
  }
}
parameters {
  cholesky_factor_corr[3] Lcorr0;
  row_vector[3] alpha[J];  // player skills over the rating periods
  real beta;  // how uncertainty grows as a player misses time
  vector<lower=0>[3] tau;
  vector[J] erratic;  // player's baseline erraticity (same across all court types)
  matrix[N_train, 2] volatility;  // time dependent volatility of the two competing players that accounts for their baseline erraticity and "time off" for each match
}
transformed parameters {
  // define 3x3 correlation and covariance matrices
  cov_matrix[3] Omega;
  corr_matrix[3] O;
  O = multiply_lower_tri_self_transpose(Lcorr0);
  Omega = quad_form_diag(O, tau);
}
model {
  // Priors
  Lcorr0 ~ lkj_corr_cholesky(1);
  tau ~ cauchy(0, 25);
  beta ~ normal(0, 1);
  alpha ~ multi_normal(Zero, Omega);
  erratic ~ normal(0, 1);
  // Likelihood
  // Build up volatilities for each match
  volatility[, 1] ~ normal(0, exp(beta .* X1 + z1 * erratic));
  volatility[, 2] ~ normal(0, exp(beta .* X2 + z2 * erratic));
  // Build up set win probability vector for each match
  vector[N_train] s;
  s = inv_logit(X_skill * to_vector(to_matrix(alpha)) + volatility[, 1] - volatility[, 2]);
  target += negbin_vec_lpmf(winner_y | s, r);
}
