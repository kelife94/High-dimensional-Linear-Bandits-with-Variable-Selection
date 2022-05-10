library(glmnet)
library(MASS)
library(picasso)
library(foreach)
library(doParallel)
library(ncvreg)

print('Experiment for scad ucb with variable selection and ridge regression')

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
ridge0 = 5

# parameter for additional l2-penalty
# alpha = 0.5 in the code correpsonds to alpha=1 in the paper
# alpha = 0.1 in the code correpsonds to alpha=9 in the paper
# alpha = 0.9 in the code correpsonds to alpha=1/9 in the paper
alpha = 0.1

# time point of cross-validation
num_epoch = 4
cv_times = c(0, sapply(seq(num_epoch - 1, 1, by=-1), function(x) {Horizon %/% (2 ^ x)}))
cv_times = unique(cv_times)

# initial parameters for algorithm
tau0_list = c(1, 5) * 10 ^ rep(-4:0, each=2)
params_list = data.matrix(tau0_list)

nfolds = 3
lambda0_list = c(1, 5) * 10 ^ rep(-6:0, each=2)
alpha_list = seq(0.1, 0.9, by=0.1)
cv_params_list = data.matrix(expand.grid(lambda0=lambda0_list, alpha=alpha_list))

print('Start running!')

cl = makeCluster(10)
registerDoParallel(cl)

comb = function(...){
  mapply(rbind, ..., SIMPLIFY = FALSE)
}

time1 = proc.time()

paral = foreach(idx = 1:nrow(params_list), .combine = comb, .packages = c('MASS', 'picasso', 'glmnet', 'ncvreg'))

result = paral%dopar%{
  
  print(paste('Running core:', idx))
  
  tau0 = params_list[idx, 1]
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
    
    ridge_val = NULL
    lambda1_val = NULL
    
    ridge0 = 5
    alpha = 0.1
    
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
        
        if ((t - 1 - init) == 0){
          E = matrix(0, nrow=nrow(Xmat), ncol=length(lambda0_list))
          fold = sample(1:nrow(Xmat) %% nfolds)
          fold[fold == 0] = nfolds
          for (i in 1:length(lambda0_list)){
            lambda = lambda0_list[i] / alpha * sqrt((log(t-1) + log(d)) / (t-1))
            for (k in 1:nfolds){
              cv_fit = ncvreg(Xmat[fold!=k, ], Ymat[fold!=k], standardize=F, penalty="SCAD", lambda=lambda, alpha=alpha, intercept=T)
              yhat = predict(cv_fit, Xmat[fold==k, ], type="response")
              loss = (Ymat[fold==k] - yhat) ^ 2
              E[fold==k, i] = loss
            }
          }
          cve = apply(E, 2, mean)
          min = which.min(cve)
          lambda0 = lambda0_list[min] / alpha
          print(paste("Optimal Lambda0:", lambda0_list[min], "alpha:", alpha, "MSE:", min(cve)))
        }
        
        if ((t - 1 - init) %in% cv_times[-1]){
          
          E = matrix(0, nrow=nrow(Xmat), ncol=nrow(cv_params_list))
          fold = sample(1:nrow(Xmat) %% nfolds)
          fold[fold == 0] = nfolds
          for (i in 1:nrow(cv_params_list)){
            lambda = cv_params_list[i, 1] / cv_params_list[i, 2] * sqrt((log(t-1) + log(d)) / (t-1))
            for (k in 1:nfolds){
              cv_fit = ncvreg(Xmat[fold!=k, ], Ymat[fold!=k], standardize=F, penalty="SCAD", lambda=lambda, alpha=cv_params_list[i, 2], intercept=T)
              yhat = predict(cv_fit, Xmat[fold==k, ], type="response")
              loss = (Ymat[fold==k] - yhat) ^ 2
              E[fold==k, i] = loss
            }
          }
          cve = apply(E, 2, mean)
          min = which.min(cve)
          alpha = cv_params_list[min, 2]
          lambda0 = cv_params_list[min, 1] / cv_params_list[min, 2]
          print(paste("Optimal Lambda0:", cv_params_list[min, 1], "alpha:", alpha, "MSE:", min(cve)))
        }
        
        lambda1 = lambda0 * sqrt((log(t-1) + log(d)) / (t-1))
        mod = ncvreg(Xmat, Ymat, standardize=F, penalty="SCAD", lambda=lambda1, alpha=alpha, intercept=T)
        active_set = which(coef(mod)[-1] != 0)
        
        lambda1_val = c(lambda1_val, lambda1)
        
        if (length(active_set) > 1){
          
          if ((t - 1 - init) %in% cv_times){
            cv.mod2 = cv.glmnet(Xmat[, active_set], Ymat, alpha=0, standardize=FALSE, nfold=3)
            ridge0 = cv.mod2$lambda.min * sqrt(t - 1)
            print(paste("Time:", t, "Ridge0:", ridge0))
          }
          
          ridge1 = ridge0 / sqrt(t - 1)
          ridge_val = rbind(ridge_val, ridge1)
          mod2 = glmnet(Xmat[, active_set], Ymat, alpha=0, standardize=FALSE, lambda=ridge1)
          tau1 = tau0 * sqrt(length(active_set) * (log(length(active_set)) + log(t-1)) / (t-1))
          beta2 = coef(mod2)
          beta2[is.na(beta2)] = 0
          
          # weighted norm for each arm feature \|x_{a,t}\|_{\Sigma_t}
          design = cbind(1, Xmat[, active_set])
          svd_res = svd(design)
          weighted_norm = function(x){
            
            d_sq = svd_res$d ^ 2 / (t - 1)
            weight = d_sq / (d_sq + ridge1) ^ 2
            vector = as.vector(t(svd_res$v) %*% x)
            val = sqrt(sum(vector ^ 2 * weight))
            return (val)
          }
          
          max_arm = 1
          max_pi_value = - Inf
          for (i in 1:K){
            
            feature = c(1, x_values[[i]][t, active_set]) # i-th observation feature at time t
            val = sum(feature * beta2) + tau1 * weighted_norm(feature)
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
      
      if (t %% 10 == 0){
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
save(params_list, result, file="scad_ucb_vs_ridge_cv.RData")

time2 = proc.time()
print(time2 - time1)

print('Job is done!')

# find the optimal initial parameters
scad_ucb_vs_ridge_cv = result$regret
scad_ucb_vs_ridge_cv_mat = scad_ucb_vs_ridge_cv[, Horizon]
scad_ucb_vs_ridge_cv_mat
params_list[which.min(scad_ucb_vs_ridge_cv[, Horizon]), ]
min(scad_ucb_vs_ridge_cv[, Horizon])



