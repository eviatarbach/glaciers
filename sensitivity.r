library("sensitivity")
library("mcmc")

dyn.load("/home/eviatar/anaconda3/pkgs/python-3.6.1-0/lib/libpython3.6m.so.1.0")
library("rPython")

sens_glacier <- function(param_vals){
  return(python.call("sample.sens_glacier", param_vals))
}

Xset <- function(n, Sj, Sjc, xjc){
  python.assign("Sj", Sj)
  python.assign("Sjc", Sjc)
  python.assign("xjc", xjc)
  python.assign("n", n)
  if (is.null(Sjc)){
    python.exec("res = sample.sample_joint(n, Sj)")
    return(do.call(rbind, python.get("res")))
  }
  python.exec("pdf = sample.conditional_PDF(Sj, Sjc, xjc)")
  logpdf <- function(yj){
    python.assign("yj", yj)
    python.exec("res = pdf(yj)")
    return(log(python.get("res")))
  }
  out<-metrop(logpdf, rep(0, length(Sj)), nbatch=1000, scale=0.6)
  return(metrop(out, nbatch=n)$batch)
}

# Xall <- function(n){
#   logpdf <- function(yj){
#     python.assign("yj", yj)
#     python.exec("res = sample.joint_PDF(yj)")
#     return(log(python.get("res")))
#   }
#   out<-metrop(logpdf, rep(0, 7), nbatch=1000, scale=0.6)
#   return(metrop(out, nbatch=n))
# }
Xall <- function(n){
  python.assign("n", n)
  python.exec("import numpy; samples = sample.dat[numpy.random.choice(sample.dat.shape[0], n), :].tolist()")
  return(do.call(rbind, python.get("samples")))
}

python.exec("import sys; sys.path.append('/home/eviatar/code/glaciers')")

#python.load("sensitivity_analysis.py")
python.exec("import sample")
#sample(1, c(6), c(4), c(1e6))
# X <- t(do.call(cbind, python.get("X")))
# df <- read.csv("to_r.csv", header=TRUE)
#mask = !(rowSums(is.na(X)) > 0)
#X <- df[,c('G', 'zela', 'ca', 'cl', 'SLOPE_avg', 'volume', 'lapse_rate')][!(rowSums(is.na(X)) > 0),][1:10000,]
#y <- (df$sensitivity)[!(rowSums(is.na(X)) > 0)][1:10000]

# x <- sensiFdiv(model=sens_glacier, X)
