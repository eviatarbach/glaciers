library("sensitivity")

dyn.load(paste(system("conda info --root", intern=TRUE),
               "/pkgs/python-3.6.1-0/lib/libpython3.6m.so.1.0", sep=""))
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

python.assign("wd", getwd())
python.exec("import sys; sys.path.append(wd)")
python.exec("import sample")

HSIC_sens <- function(n_samples=500){
  X <- sample_sens(n_samples)
  Y <- sens_glacier(X)
  m <- Y != 0
  res <- sensiHSIC(X=X[m,], nboot=10)
  write.table(tell(res, y=log(-Y[m]))$S, "data/HSIC_sens.txt", sep=",")
}

HSIC_tau <- function(n_samples=500){
  X <- sample_tau(n_samples)
  Y <- tau_glacier(X)
  m <- Y > 0
  res <- sensiHSIC(X=X[m,], nboot=10)
  write.table(tell(res, y=log(Y[m]))$S, "data/HSIC_tau.txt", sep=",")
}

pairwise_HSIC <- function(n_samples=1000){
  X <- sample_all(n_samples)
  pairs <- combn(1:6, 2)
  HSIC_mat <- matrix(0, nrow=6, ncol=6)
  for(i in 1:dim(pairs)[2]){
    HSIC_mat[pairs[1, i], pairs[2, i]] <- tell(sensiHSIC(X=X[, c(pairs[1, i], pairs[1, i])]),
                                               y=X[, pairs[2, i]])$S$original[1]
  }
  return(HSIC_mat)
}
