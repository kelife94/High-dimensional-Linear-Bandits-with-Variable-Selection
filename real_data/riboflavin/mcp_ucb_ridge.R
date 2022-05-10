library(glmnet)
library(MASS)
library(picasso)
library(foreach)
library(doParallel)
library(ncvreg)

print("Experiment for mcp ucb without variable selection")

data = read.csv("./riboflavin.csv", row.names=1, header=TRUE)
data = t(as.matrix(data))
data = scale(data)

x_values_mat = as.matrix(data[, -1])
y_values_mat = as.vector(data[, 1])

cor_xy = as.vector(cor(x_values_mat, y_values_mat))
idx_cor = which(abs(cor_xy) >= 0.2)
x_values_mat = x_values_mat[, idx_cor]

# hyperparameters
Horizon = 100
init = 5
K = 2
d = ncol(x_values_mat)
# number of replication
num_rep = 5

# parameter for additional l2-penalty
# alpha = 0.5 in the code correpsonds to alpha=1 in the paper
# alpha = 0.1 in the code correpsonds to alpha=9 in the paper
# alpha = 0.9 in the code correpsonds to alpha=1/9 in the paper
alpha = 0.5

# initial parameters for algorithm
lambda0_list = c(1, 5) * 10 ^ rep(-6:0, each=2)
tau0_list = 10 ^ c(-3:0)
params_list = data.matrix(expand.grid(lambda0=lambda0_list, tau0=tau0_list))

print('Start running!')

cl = makeCluster(20)
registerDoParallel(cl)

comb = function(...){
  mapply(rbind, ..., SIMPLIFY = FALSE)
}

time1 = proc.time()

paral = foreach(idx = 1:nrow(params_list), .combine = comb, .packages = c('MASS', 'picasso', 'glmnet', 'ncvreg'))

result = paral%dopar%{
  
  print(paste0('Running core:', idx))
  
  lambda0 = params_list[idx, 1] / alpha
  tau0 = params_list[idx, 2]
  rep_regret = NULL
  rep_accuracy = NULL
  
  for (k_rep in 1:num_rep){
    
    set.seed(k_rep)
    x_values = list()
    y_values = list()
    
    indices = sample(1:nrow(x_values_mat), size=2*(Horizon + init), replace=T)
    idx_arm1 = indices[1:(Horizon+init)]
    idx_arm2 = indices[(Horizon+init+1):(2*(Horizon+init))]
    x_values[[1]] = x_values_mat[idx_arm1, ]
    y_values[[1]] = y_values_mat[idx_arm1]
    x_values[[2]] = x_values_mat[idx_arm2, ]
    y_values[[2]] = y_values_mat[idx_arm2]
    
    # selected design matrix and reward vectors 
    Xmat = NULL
    Ymat = NULL
    
    # cumulative regret = misclassification
    regret=NULL
    accuracy = NULL
    
    for (t in 1:init){
      
      # select the t-th observation of the t-th class
      max_arm = ifelse(t%%K,  t%%K, K)
      u = x_values[[max_arm]][t, ]
      Xmat = rbind(Xmat, u)
      Ymat = c(Ymat, y_values[[max_arm]][t])
      
    }
    
    for (t in (init+1):(init+Horizon)){
      
      if (sd(Ymat) == 0){
        max_arm = sample(1:K, size=1)
      }else {
        
        lambda1 = lambda0 * sqrt((log(t-1) + log(d)) / (t-1))
        tau1 = tau0 * sqrt((log(t-1) + log(d)) / (t-1))
        mod = ncvreg(Xmat, Ymat, standardize=F, penalty="MCP", lambda=lambda1, alpha=alpha, intercept=T)
        beta = coef(mod)[-1]
        
        max_arm = 1
        max_pi_value = - Inf
        for (i in 1:K){
          feature = x_values[[i]][t, ] # i-th observation feature at time t
          val = sum(feature * beta) + tau1 * max(abs(feature))
          if (val == max_pi_value){
            max_arm = ifelse(sample(c(TRUE, FALSE), size=1), i, max_arm) 
          }else if (val >  max_pi_value){
            max_arm = i
            max_pi_value = val
          }
        }
      }
      
      # reward of the selected arm
      Xmat = rbind(Xmat, x_values[[max_arm]][t, ])
      Ymat = c(Ymat, y_values[[max_arm]][t])
      
      # instant regret
      opt_arm = which.max(c(y_values[[1]][t], y_values[[2]][t]))
      max_reward = y_values[[opt_arm]][t]
      regret = c(regret, max_reward - y_values[[max_arm]][t])
      accuracy = c(accuracy, as.numeric(max_arm == opt_arm))
      
      if (t %% 10 == 0){
        print(paste("Time:", t, "Regret:", sum(regret), "Accuracy:", mean(accuracy)))
        print(sum(beta != 0))
      }
    }
    
    rep_regret = rbind(rep_regret, cumsum(regret))
    rep_accuracy = rbind(rep_accuracy, cumsum(accuracy) / (1:Horizon))
  }
  
  return (list(accuracy=apply(rep_accuracy, 2, mean), 
               accuracy_sd=apply(rep_accuracy, 2, sd),
               regret=apply(rep_regret, 2, mean),
               regret_sd=apply(rep_regret, 2, sd)
  ))
}

stopCluster(cl)
save(params_list, result, file="mcp_ucb_ridge.RData")

time2 = proc.time()
print(time2 - time1)

print('Job is done!')

# find the optimal initial parameters
mcp_ucb_ridge_cv = result$regret
mcp_ucb_ridge_cv_mat = matrix(mcp_ucb_ridge_cv[, Horizon], nrow=length(lambda0_list))
mcp_ucb_ridge_cv_mat
params_list[which.min(mcp_ucb_ridge_cv[, Horizon]), ]
min(mcp_ucb_ridge_cv[, Horizon])

apply(mcp_ucb_ridge_cv_mat, 1, min)


