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
  matrix[N_train, J] z;  // indicator matrix where z[i,j] = 1 if player j participate in match i
  matrix[N_train, 3*J] X_skill;  // design matrix for player skills
  matrix[N_train, J] X_skill_time;  // number of months between each player's last event and current event 
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
  beta ~ normal(0, 25);
  // NOTE: if you comment out the below `for` loop that samples the player skills `alpha`, and just use `alpha ~ multi_normal(Zero, Omega);`, 
  // the model fits fine.
  for (i in 2:(N_train+1)) {
    for (j in 1:J) {
      if (z[i-1, j] == 1 && X_skill_time[i-1, j] > 0) { // Time-varying covariance for the two players participating in match i-1
        alpha[j] ~ multi_normal(Zero, quad_form_diag(O, tau + beta^2*X_skill_time[i-1, j]));
      } else { // Otherwise use constant covariance
        alpha[j] ~ multi_normal(Zero, Omega);
      }
    }
  }
  // Likelihood
  // Build up set win probability vector for each match
  vector[N_train] s;
  s = inv_logit(X_skill * to_vector(to_matrix(alpha)));
  target += negbin_vec_lpmf(winner_y | s, r);
}
