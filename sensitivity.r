library("sensitivity")
library("mcmc")

dyn.load("/home/eviatar/anaconda3/pkgs/python-3.6.1-0/lib/libpython3.6m.so.1.0")
library("rPython")

sens_glacier <- function(param_vals){
  return(python.call("sample.sens_glacier", param_vals))
}

tau_glacier <- function(param_vals){
  return(python.call("sample.tau_glacier", param_vals))
}

Xset_sens <- function(n, Sj, Sjc, xjc){
  python.assign("Sj", Sj)
  python.assign("Sjc", Sjc)
  python.assign("xjc", xjc)
  python.assign("n", n)
  if (is.null(Sjc)){
    python.exec("res = sample.sample_joint_sens(n, Sj)")
    return(do.call(rbind, python.get("res")))
  }
  python.exec("pdf = sample.conditional_PDF_sens(Sj, Sjc, xjc)")
  logpdf <- function(yj){
    python.assign("yj", yj)
    python.exec("res = pdf(yj)")
    return(log(python.get("res")))
  }
  # Get into high-probability region
  out<-metrop(logpdf, rep(0, length(Sj)), nbatch=50, scale=0.6)

  return(metrop(out, nbatch=n)$batch)
}

Xall_sens <- function(n){
  python.assign("n", n)
  python.exec("res = sample.sample_joint_sens(n)")
  return(do.call(rbind, python.get("res")))
}

Xset_tau <- function(n, Sj, Sjc, xjc){
  python.assign("Sj", Sj)
  python.assign("Sjc", Sjc)
  python.assign("xjc", xjc)
  python.assign("n", n)
  if (is.null(Sjc)){
    python.exec("res = sample.sample_joint_tau(n, Sj)")
    return(do.call(rbind, python.get("res")))
  }
  python.exec("pdf = sample.conditional_PDF_tau(Sj, Sjc, xjc)")
  logpdf <- function(yj){
    python.assign("yj", yj)
    python.exec("res = pdf(yj)")
    return(log(python.get("res")))
  }
  # Get into high-probability region
  out<-metrop(logpdf, rep(0, length(Sj)), nbatch=50, scale=0.6)
  
  return(metrop(out, nbatch=n)$batch)
}

Xall_tau <- function(n){
  python.assign("n", n)
  python.exec("res = sample.sample_joint_tau(n)")
  return(do.call(rbind, python.get("res")))
}

python.exec("import sys; sys.path.append('/home/eviatar/code/glaciers')")
python.exec("import sample")