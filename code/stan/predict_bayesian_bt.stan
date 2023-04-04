functions {
  // Compute mu parameter for Stan's paramaterization of negative binomial
  vector get_mu(int[] phi, vector pwinner) {
    vector[num_elements(phi)] out;
    out = to_vector(phi) .* ((1 - pwinner) ./ pwinner);
    //for (i in 1:num_elements(phi)) {
      //out[i] = phi[i] * (1 - pwinner[i]) / pwinner[i];
    //}
    return out;
  }
  // Compute set win probability given win probability of game g
  vector get_set_win_prob(vector g) {
    vector[rows(g)] s = g^6  // win 6-0
     + choose(6, 5) .* g^5 .* (1-g) .* g  // win 6-1
     + choose(7, 5) .* g^5 .* (1-g)^2 .* g  // win 6-2
     + choose(8, 5) .* g^5 .* (1-g)^3 .* g  // win 6-3
     + choose(9, 5) .* g^5 .* (1-g)^4 .* g // win 6-4
     + choose(10, 5) .* g^5 .* (1-g)^5 .* g^2  // win 7-5
     + choose(10, 5) .* g^5 .* (1-g)^5 .* choose(2, 1) .* g .* (1-g) .* g;  // win 7-6
     return s;
  }
  // Compute match win probability given set win probability s AND best of n (either 3 or 5)
  vector get_match_win_prob(vector s, vector n) {
    vector[num_elements(s)] m;
    for (i in 1:num_elements(s)) {
      if (n[i] == 3) {
        m[i] = s[i]^2 + 2*s[i]^2*(1-s[i]);
      } else {
        m[i] = s[i]^3*(1 + 3*(1-s[i]) + 6*(1-s[i])^2);
      }
    return m;
  }
  // Computes the log-PDF of the negative binomial distribution with y failures until r successes with success probability p
  real negbin_lpdf(real[] y, vector p, vector r) {
    return sum(lchoose(to_vector(y) + r - 1, to_vector(y)) + r.*log(p) + to_vector(y).*log1m(p));
  }
}
data {
  int<lower=0> J;  // number of players
  int<lower=0> N_posterior_samples;  // number of posterior samples
  row_vector[3] alpha[J];  // player skills
  real beta;  // time off coefficient
  int<lower=3, upper=5> best_of;
}
parameters {
}
model {
}
generated quantities {
  vector[N_test] p1_tilde;  // probability player 1 wins match
  p1_tilde = get_match_win_prob(get_set_win_prob(inv_logit(X_skill_tilde*alpha_vector + (X1_tilde - X2_tilde)*beta)), best_of);  // probability of match victory for player 1
}
