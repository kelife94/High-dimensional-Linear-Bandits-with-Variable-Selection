library(glmnet)
library(MASS)
library(picasso)
library(foreach)
library(doParallel)

T=2000
K=5
d=100
s=5
rep_num = 5
sigma=1
rho = 0.1

# nonzero beta uniformly from (0.5, 1)
set.seed(1)
beta0 = rep(0,d)
S = 1:s
beta0[S] = runif(s, 0.5, 1.0)

# function for the arms covariates truncated normal with covariance 
Sigma0=matrix(0, d, d)
for(i in 1:d){
  for(j in 1:d){
    Sigma0[i,j]= rho ^ (abs(i-j))
  }
}

user=function()
{
  mat=matrix(0, nrow=d, ncol=K)
  for(j in 1:K){
    v=as.vector(mvrnorm(1, rep(0,d), Sigma=Sigma0))
    v=sapply(v, function(x) sign(x)*min(abs(x), 5))
    mat[,j]=v
  }
  return(mat)
}

# initial parameters for algorithm
lambda0_list = seq(0.5, 1.5, 0.1)
tau0_list = seq(0.1, 1, by=0.1)
params_list = data.matrix(expand.grid(lambda0=lambda0_list, tau0=tau0_list))

cl = makeCluster(22)
registerDoParallel(cl)

time1 = proc.time()

paral = foreach(idx = 1:nrow(params_list), .combine = 'rbind', .packages = c('MASS', 'picasso', 'glmnet'))

result = paral %dopar% {
  
  set.seed(19)
  lambda0 = params_list[idx, 1]
  tau0 = params_list[idx, 2]
  rep_regret = NULL
  for(rep_i in 1:rep_num){

    index=NULL
    index_opt=NULL

    # selected design matrix and reward vectors 
    Xmat = NULL
    Ymat = NULL
    
    # instant regret
    regret = NULL
    
    for(t in 1:K){
      u=user()
      eta = rnorm(n=1, mean=0, sd=sigma)
      
      Xmat = rbind(Xmat, u[, t])
      Ymat = c(Ymat, sum(beta0 * u[, t]) + eta)
      
      opt_arm = which.max(apply(u,2, function(x) sum(x*beta0)))
      index_opt = c(index_opt, opt_arm) 
      
      # regret
      index = c(index, t)
      regret = c(regret, sum(beta0 * (u[, opt_arm]-u[, t])))
    }
    
    for(t in (K+1):T){
      u=user()
      eta = rnorm(n=1, mean=0, sd=sigma)
      
      # select the best arm
      lambda1 = lambda0 * sqrt((log(t-1) + log(d)) / (t-1))
      tau1 = tau0 * sqrt((log(t-1) + log(d)) / (t-1))
      mod = picasso(Xmat, Ymat, method="mcp", lambda=lambda1, intercept=FALSE, standardize = FALSE)
      beta = mod$beta
      
      max_arm = which.max(apply(u, 2, function(x) {sum(x * beta) + tau1 * max(abs(x))}))
      index = c(index, max_arm)
      
      # optimal arm
      opt_arm = which.max(apply(u, 2, function(x) sum(x * beta0)))
      index_opt = c(index_opt, opt_arm)
      
      # reward of the selected arm
      Xmat = rbind(Xmat, u[, max_arm])
      Ymat = c(Ymat, sum(u[, max_arm] * beta0) + eta)
      
      # instant regret
      regret=c(regret, sum(beta0 * (u[, opt_arm] - u[, max_arm])))
      
      if(t %% 100 ==0){
        print(paste("Time:", t, "Regret:", sum(regret)))
      }
    }
    rep_regret = rbind(rep_regret, cumsum(regret))
  }
  
  return(apply(rep_regret, 2, mean))
}

stopCluster(cl)
save(params_list, result, file="mcp_ucb_cv.RData")


time2 = proc.time()
time2 - time1

# find the optimal initial parameters
mcp_ucb_cv = result
params_list[which.min(mcp_ucb_cv[, T]), ]
min(mcp_ucb_cv[, T])

# save the optimal cumulative regret result of all methods
regret_dataset = data.frame(mcp_ucb_cv = mcp_ucb_cv[which.min(mcp_ucb_cv[, T]), ],
                            mcp_ucb_vs_cv = mcp_ucb_vs_cv[which.min(mcp_ucb_vs_cv[, T]), ],
                            scad_ucb_cv = scad_ucb_cv[which.min(scad_ucb_cv[, T]), ],
                            scad_ucb_vs_cv = scad_ucb_vs_cv[which.min(scad_ucb_vs_cv[, T]), ])

write.csv(regret_dataset, file='./d100_s5_k5_regret.csv', row.names=FALSE)


