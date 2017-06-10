library("sensitivity")

dyn.load("/home/eviatar/anaconda3/pkgs/python-3.6.1-0/lib/libpython3.6m.so.1.0")
library("rPython")

sens_glacier <- function(param_vals){
  return(python.call("sample.sens_glacier", param_vals))
}

tau_glacier <- function(param_vals){
  return(python.call("sample.tau_glacier", param_vals))
}

sample_sens <- function(n){
  python.assign("n", n)
  python.exec("res = sample.sample_joint_sens(n)")
  return(do.call(rbind, python.get("res")))
}

sample_tau <- function(n){
  python.assign("n", n)
  python.exec("res = sample.sample_joint_tau(n)")
  return(do.call(rbind, python.get("res")))
}

sample_all <- function(n){
  python.assign("n", n)
  python.exec("res = sample.sample_joint_all(n)")
  return(do.call(rbind, python.get("res")))
}

python.exec("import sys; sys.path.append('/home/eviatar/code/glaciers')")
python.exec("import sample")

ranking_sens <- function(n_samples=500, n_trials=100){
  ind <- matrix(0, nrow=n_trials, ncol=7)
  for(i in 1:n_trials){
    X <- sample_sens(n_samples)
    Y <- sens_glacier(X)
    m <- Y != 0
    res <- sensiHSIC(X=X[m,])
    ind[i,] <- tell(res, y=Y[m])$S$original
  }
  freqs <- table(apply(ind, 1, function(x) paste(order(x), collapse="")))
  corr_mat <- cor(t(ind), method="kendall")
  corr <- mean(corr_mat[lower.tri(corr_mat, diag=FALSE)])
  return(list("freqs" = freqs, "corr" = corr))
}

ranking_tau <- function(n_samples=500, n_trials=100){
  ind <- matrix(0, nrow=n_trials, ncol=7)
  for(i in 1:n_trials){
    X <- sample_tau(n_samples)
    Y <- tau_glacier(X)
    m <- Y != 0
    res <- sensiHSIC(X=X[m,])
    ind[i,] <- tell(res, y=Y[m])$S$original
  }
  freqs <- table(apply(ind, 1, function(x) paste(order(x), collapse="")))
  corr_mat <- cor(t(ind), method="kendall")
  corr <- mean(corr_mat[lower.tri(corr_mat, diag=FALSE)])
  return(list("freqs" = freqs, "corr" = corr))
}

pairwise_HSIC <- function(n_samples=1000){
  X <- sample_all(n_samples)
  pairs <- combn(1:9, 2)
  HSIC_mat <- matrix(0, nrow=9, ncol=9)
  for(i in 1:dim(pairs)[2]){
    HSIC_mat[pairs[1, i], pairs[2, i]] <- tell(sensiHSIC(X=X[, c(pairs[1, i], pairs[1, i])]),
                                               y=X[, pairs[2, i]])$S$original[1]
  }
  return(HSIC_mat)
}