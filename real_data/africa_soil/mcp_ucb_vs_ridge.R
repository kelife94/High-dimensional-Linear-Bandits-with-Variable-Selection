library(glmnet)
library(MASS)
library(picasso)
library(foreach)
library(doParallel)
library(ncvreg)

print("Experiment for mcp ucb without variable selection")

train = read.csv("./training.csv")
train$Depth = with(train, ifelse(Depth == 'Subsoil', 0, 1))

MIR_measurements = train[, 2:2655]
MIR_DER = MIR_measurements - cbind(NA, MIR_measurements)[, -(dim(MIR_measurements)[2]+1)]
# remove the Depth (3595) columns for construction of arms
x_values_mat = cbind(train[, 3580:3594], MIR_DER[, -1])
MIR_measurements = train[, 2671:3579]
MIR_DER = MIR_measurements - cbind(NA, MIR_measurements)[, -(dim(MIR_measurements)[2]+1)]
x_values_mat = cbind(x_values_mat, MIR_DER[, -1])
x_values_mat = as.matrix(x_values_mat)
x_values_mat = scale(x_values_mat)

y_values_mat = as.vector(train$Ca) 

x_values_list = list()
y_values_list = list()
x_values_list[[1]] = x_values_mat[train$Depth == 0, ]
y_values_list[[1]] = y_values_mat[train$Depth == 0]
x_values_list[[2]] = x_values_mat[train$Depth == 1, ]
y_values_list[[2]] = y_values_mat[train$Depth == 1]

# hyperparameters
Horizon = 500
init = 70
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
lambda0_list = c(0.01, 0.05, seq(0.1, 1.2, by=0.1))
tau0_list = 10 ^ c(-5:-1)
ridge0_list = c(0.1, 0.5, 1, 5, 10)
params_list = data.matrix(expand.grid(lambda0=lambda0_list, tau0=tau0_list, ridge0=ridge0_list))

print('Start running!')

cl = makeCluster(20)
registerDoParallel(cl)

comb = function(...){
  mapply(rbind, ..., SIMPLIFY = FALSE)
}

time1 = proc.time()

paral = foreach(idx = 1:nrow(params_list), .combine = comb, .packages = c('MASS', 'picasso', 'glmnet', 'ncvreg'))

result = paral%dopar%{
  
  print(paste('Running core:', idx))
  
  lambda0 = params_list[idx, 1] / alpha
  tau0 = params_list[idx, 2]
  ridge0 = params_list[idx, 3]
  rep_regret = NULL
  rep_accuracy = NULL
  
  for (k_rep in 1:num_rep){
    
    set.seed(k_rep)
    x_values = list()
    y_values = list()
    for(i in 1:K){
      indices = sample(1: nrow(x_values_list[[i]]), size=Horizon+init, replace=F)
      x_values[[i]] = x_values_list[[i]][indices, ]
      y_values[[i]] = y_values_list[[i]][indices]
    }
    
    # selected design matrix and reward vectors 
    Xmat = NULL
    Ymat = NULL
    
    # cumulative regret = misclassification
    regret=NULL
    accuracy = NULL
    
    for (t in 1:init){
      
      # random sample for initial period
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
        mod = ncvreg(Xmat, Ymat, standardize=F, penalty="MCP", lambda=lambda1, alpha=alpha, intercept=T)
        active_set = which(coef(mod)[-1] != 0)
        
        if (length(active_set) > 1){
          
          ridge1 = ridge0 / sqrt(t - 1)
          mod2 = glmnet(Xmat[, active_set], Ymat, alpha=0, standardize=FALSE, lambda=ridge1)
          tau1 = tau0 * sqrt(length(active_set) * (log(length(active_set)) + log(t-1)) / (t-1))
          beta2 = coef(mod2)
          beta2[is.na(beta2)] = 0
          
          # weighted norm for each arm feature \|x_{a,t}\|_{\Sigma_t}
          design = cbind(1, Xmat[, active_set])
          svd_res = svd(design)
          weighted_norm = function(x){
            
            weight = ((svd_res$d) ^ 2 + ridge1) ^ 2 / ((t - 1) * svd_res$d ^ 2)
            vector = as.vector(t(svd_res$v) %*% x)
            val = sqrt(sum(vector ^ 2 / weight))
            return (val)
          }
          
          max_arm = 1
          max_pi_value = - Inf
          for (i in 1:K){
            
            feature = c(1, x_values[[i]][t, active_set]) # i-th observation feature at time t
            val = sum(feature * beta2) + tau1 * weighted_norm(feature)
            if (t %% 100 == 0){
              print(paste("Estimated Reward:", sum(feature * beta2)))
              print(paste("Confidence Bound:", tau1 * weighted_norm(feature)))
            }
            if (val == max_pi_value){
              max_arm = ifelse(sample(c(TRUE, FALSE), size=1), i, max_arm) 
            }else if (val >  max_pi_value){
              max_arm = i
              max_pi_value = val
            }
          }
        }else {
          max_arm = sample(1:K, size=1)
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
      
      if (t %% 100 == 0){
        print(paste("Time:", t, "Regret:", sum(regret), "Accuracy:", mean(accuracy)))
        print(length(active_set))
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
save(params_list, result, file="mcp_ucb_vs_ridge.RData")

time2 = proc.time()
print(time2 - time1)

print('Job is done!')

# find the optimal initial parameters
mcp_ucb_vs_ridge_cv = result$regret
mcp_ucb_vs_ridge_cv_mat = array(mcp_ucb_vs_ridge_cv[, Horizon], 
                                 dim=c(length(lambda0_list), length(tau0_list), length(ridge0_list)))
params_list[which.min(mcp_ucb_vs_ridge_cv[, Horizon]), ]
min(mcp_ucb_vs_ridge_cv[, Horizon])

apply(mcp_ucb_vs_ridge_cv_mat, 1, min)


