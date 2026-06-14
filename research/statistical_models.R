# QUANTUM-FORGE Statistical Models
# Advanced statistical modeling for quantitative finance
# Author: QUANTUM-FORGE Research Team
# Last Updated: 2024

# Load required libraries
suppressPackageStartupMessages({
  library(quantmod)
  library(PerformanceAnalytics)
  library(rugarch)
  library(fGarch)
  library(rmgarch)
  library(copula)
  library(VineCopula)
  library(vars)
  library(urca)
  library(forecast)
  library(changepoint)
  library(bcp)
  library(HiddenMarkov)
  library(depmixS4)
  library(extremevalues)
  library(evd)
  library(POT)
  library(xts)
  library(zoo)
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(corrplot)
  library(RcppArmadillo)
  library(parallel)
  library(doParallel)
})

# Initialize parallel processing
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

cat("QUANTUM-FORGE Statistical Models Initialized\n")
cat("Using", n_cores, "CPU cores for parallel processing\n")

#' Advanced GARCH Model Framework
#' 
#' Implements multiple GARCH variants with regime switching and asymmetric effects
#' @param returns xts object of return series
#' @param model_type character: "sGARCH", "eGARCH", "gjrGARCH", "apARCH"
#' @param distribution character: "norm", "std", "sstd", "ged", "jsu"
#' @param regime_switching logical: enable regime switching
#' @return list containing model results and forecasts
advanced_garch_modeling <- function(returns, 
                                  model_type = "eGARCH", 
                                  distribution = "sstd",
                                  regime_switching = TRUE) {
  
  cat("Fitting advanced GARCH model:", model_type, "with", distribution, "distribution\n")
  
  # Basic GARCH specification
  if (model_type == "eGARCH") {
    spec <- ugarchspec(
      variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
      distribution.model = distribution
    )
  } else if (model_type == "gjrGARCH") {
    spec <- ugarchspec(
      variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
      distribution.model = distribution
    )
  } else if (model_type == "apARCH") {
    spec <- ugarchspec(
      variance.model = list(model = "apARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
      distribution.model = distribution
    )
  } else {
    spec <- ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
      distribution.model = distribution
    )
  }
  
  # Fit the model
  fit <- ugarchfit(spec, returns, solver = "hybrid")
  
  # Model diagnostics
  residuals <- residuals(fit, standardize = TRUE)
  ljung_box <- Box.test(residuals^2, lag = 10, type = "Ljung-Box")
  arch_test <- ArchTest(residuals, lags = 5)
  
  # Forecasting
  forecast_1d <- ugarchforecast(fit, n.ahead = 1)
  forecast_10d <- ugarchforecast(fit, n.ahead = 10)
  
  # Value at Risk calculations
  var_95 <- quantile(residuals, 0.05) * sigma(fit)[length(sigma(fit))]
  var_99 <- quantile(residuals, 0.01) * sigma(fit)[length(sigma(fit))]
  
  # Expected Shortfall
  es_95 <- mean(residuals[residuals <= quantile(residuals, 0.05)]) * sigma(fit)[length(sigma(fit))]
  es_99 <- mean(residuals[residuals <= quantile(residuals, 0.01)]) * sigma(fit)[length(sigma(fit))]
  
  results <- list(
    model = fit,
    residuals = residuals,
    diagnostics = list(
      ljung_box = ljung_box,
      arch_test = arch_test,
      aic = infocriteria(fit)[1],
      bic = infocriteria(fit)[2]
    ),
    forecasts = list(
      one_day = forecast_1d,
      ten_day = forecast_10d
    ),
    risk_metrics = list(
      var_95 = var_95,
      var_99 = var_99,
      es_95 = es_95,
      es_99 = es_99,
      current_volatility = sigma(fit)[length(sigma(fit))]
    )
  )
  
  cat("GARCH model fitted successfully. AIC:", results$diagnostics$aic, "\n")
  return(results)
}

#' Multivariate GARCH (DCC) Model
#' 
#' Dynamic Conditional Correlation model for portfolio risk management
#' @param returns matrix or xts of return series
#' @param dcc_order vector: DCC order parameters
#' @return DCC model results with correlation forecasts
multivariate_garch_dcc <- function(returns, dcc_order = c(1, 1)) {
  
  cat("Fitting DCC-GARCH model for", ncol(returns), "assets\n")
  
  # Individual GARCH specifications
  uspec <- multispec(replicate(ncol(returns), 
                              ugarchspec(mean.model = list(armaOrder = c(1, 1)), 
                                        variance.model = list(garchOrder = c(1, 1)))))
  
  # DCC specification
  dcc_spec <- dccspec(uspec, dccOrder = dcc_order, distribution = "mvnorm")
  
  # Fit DCC model
  dcc_fit <- dccfit(dcc_spec, returns)
  
  # Extract dynamic correlations
  dyn_corr <- rcor(dcc_fit)
  
  # Calculate portfolio metrics
  n_assets <- ncol(returns)
  equal_weights <- rep(1/n_assets, n_assets)
  
  # Portfolio volatility over time
  portfolio_vol <- sapply(1:dim(dyn_corr)[3], function(i) {
    sqrt(t(equal_weights) %*% dyn_corr[,,i] %*% diag(sigma(dcc_fit)[i,]) %*% equal_weights)
  })
  
  # Forecast correlations
  dcc_forecast <- dccforecast(dcc_fit, n.ahead = 10)
  
  results <- list(
    model = dcc_fit,
    dynamic_correlations = dyn_corr,
    portfolio_volatility = portfolio_vol,
    forecast = dcc_forecast,
    average_correlation = apply(dyn_corr, 3, function(x) mean(x[upper.tri(x)])),
    diagnostics = list(
      likelihood = likelihood(dcc_fit),
      aic = infocriteria(dcc_fit)[1],
      bic = infocriteria(dcc_fit)[2]
    )
  )
  
  cat("DCC-GARCH model fitted. Average correlation:", 
      round(mean(results$average_correlation), 4), "\n")
  
  return(results)
}

#' Regime Switching Models
#' 
#' Hidden Markov Models for market regime detection
#' @param returns xts object of returns
#' @param n_regimes integer: number of regimes
#' @param include_vol logical: include volatility regime switching
#' @return HMM results with regime probabilities
regime_switching_model <- function(returns, n_regimes = 2, include_vol = TRUE) {
  
  cat("Fitting", n_regimes, "regime Hidden Markov Model\n")
  
  # Prepare data
  ret_data <- as.numeric(returns)
  ret_data <- ret_data[!is.na(ret_data)]
  
  if (include_vol) {
    # Include realized volatility as additional observable
    vol_data <- runSD(returns, n = 20)
    vol_data <- as.numeric(vol_data)
    vol_data <- vol_data[!is.na(vol_data)]
    
    # Align lengths
    min_length <- min(length(ret_data), length(vol_data))
    obs_data <- cbind(ret_data[(length(ret_data) - min_length + 1):length(ret_data)],
                      vol_data[(length(vol_data) - min_length + 1):length(vol_data)])
  } else {
    obs_data <- matrix(ret_data, ncol = 1)
  }
  
  # Fit HMM using depmixS4
  if (include_vol) {
    hmm_model <- depmix(list(ret_data ~ 1, vol_data ~ 1), 
                        data = data.frame(ret_data = obs_data[,1], 
                                         vol_data = obs_data[,2]),
                        nstates = n_regimes,
                        family = list(gaussian(), gaussian()))
  } else {
    hmm_model <- depmix(ret_data ~ 1, 
                        data = data.frame(ret_data = obs_data[,1]),
                        nstates = n_regimes,
                        family = gaussian())
  }
  
  # Fit the model
  hmm_fit <- fit(hmm_model)
  
  # Extract regime probabilities
  regime_probs <- posterior(hmm_fit)
  
  # Calculate regime statistics
  regime_stats <- list()
  for (i in 1:n_regimes) {
    regime_periods <- which(regime_probs$state == i)
    if (length(regime_periods) > 0) {
      regime_stats[[paste0("regime_", i)]] <- list(
        mean_return = mean(obs_data[regime_periods, 1]),
        volatility = sd(obs_data[regime_periods, 1]),
        duration = length(regime_periods),
        probability = sum(regime_probs[,i+1]) / nrow(regime_probs)
      )
    }
  }
  
  # Regime transition matrix
  transition_matrix <- getpars(hmm_fit)[1:(n_regimes^2)]
  transition_matrix <- matrix(transition_matrix, nrow = n_regimes, byrow = TRUE)
  
  results <- list(
    model = hmm_fit,
    regime_probabilities = regime_probs,
    regime_statistics = regime_stats,
    transition_matrix = transition_matrix,
    current_regime = regime_probs$state[nrow(regime_probs)],
    current_probability = max(regime_probs[nrow(regime_probs), -1]),
    diagnostics = list(
      aic = AIC(hmm_fit),
      bic = BIC(hmm_fit),
      loglik = logLik(hmm_fit)
    )
  )
  
  cat("HMM fitted. Current regime:", results$current_regime, 
      "with probability:", round(results$current_probability, 3), "\n")
  
  return(results)
}

#' Factor Models and Risk Attribution
#' 
#' Multi-factor models for risk and return attribution
#' @param asset_returns xts object of asset returns
#' @param factor_returns xts object of factor returns
#' @param model_type character: "linear", "time_varying", "regime_switching"
#' @return factor model results with risk attribution
factor_model_analysis <- function(asset_returns, factor_returns, model_type = "time_varying") {
  
  cat("Fitting", model_type, "factor model\n")
  
  # Align data
  common_dates <- intersect(index(asset_returns), index(factor_returns))
  asset_data <- asset_returns[common_dates]
  factor_data <- factor_returns[common_dates]
  
  if (model_type == "linear") {
    # Standard linear factor model
    model <- lm(asset_data ~ factor_data)
    
    # Calculate factor loadings
    factor_loadings <- coef(model)[-1]  # Exclude intercept
    
    # Risk attribution
    factor_var <- var(factor_data)
    specific_var <- var(residuals(model))
    total_var <- var(asset_data, na.rm = TRUE)
    
    factor_contribution <- as.numeric(t(factor_loadings) %*% factor_var %*% factor_loadings)
    r_squared <- 1 - specific_var / total_var
    
  } else if (model_type == "time_varying") {
    # Kalman filter for time-varying loadings
    library(dlm)
    
    # Set up state space model
    build_tv_factor <- function(pars) {
      dlmModReg(factor_data, dV = exp(pars[1]), dW = exp(pars[2:length(pars)]))
    }
    
    # Estimate parameters
    mle_fit <- dlmMLE(asset_data, pars = rep(0, ncol(factor_data) + 1), build_tv_factor)
    tv_model <- build_tv_factor(mle_fit$par)
    
    # Filter and smooth
    tv_filter <- dlmFilter(asset_data, tv_model)
    tv_smooth <- dlmSmooth(tv_filter)
    
    # Time-varying factor loadings
    factor_loadings <- tv_smooth$s[-1, -1]  # Remove first row and intercept
    
    # Current loadings
    current_loadings <- factor_loadings[nrow(factor_loadings), ]
    
    r_squared <- 1 - var(residuals(tv_filter)) / var(asset_data, na.rm = TRUE)
    
  }
  
  results <- list(
    model = if(model_type == "linear") model else tv_model,
    factor_loadings = if(model_type == "time_varying") current_loadings else factor_loadings,
    r_squared = r_squared,
    diagnostics = list(
      adj_r_squared = if(model_type == "linear") summary(model)$adj.r.squared else NA,
      residual_tests = if(model_type == "linear") {
        list(
          jarque_bera = jarque.bera.test(residuals(model)),
          ljung_box = Box.test(residuals(model), type = "Ljung-Box")
        )
      } else NA
    )
  )
  
  if (model_type == "time_varying") {
    results$time_varying_loadings <- factor_loadings
    results$filtered_states <- tv_filter
    results$smoothed_states <- tv_smooth
  }
  
  cat("Factor model fitted. R-squared:", round(r_squared, 3), "\n")
  return(results)
}

#' Jump Diffusion Models
#' 
#' Merton jump diffusion model for capturing extreme price movements
#' @param returns xts object of returns
#' @param method character: estimation method
#' @return jump diffusion parameters and jump detection
jump_diffusion_model <- function(returns, method = "mle") {
  
  cat("Fitting Merton Jump Diffusion Model\n")
  
  # Convert to numeric
  ret_data <- as.numeric(returns)
  ret_data <- ret_data[!is.na(ret_data)]
  
  # Merton Jump Diffusion likelihood function
  merton_likelihood <- function(params) {
    mu <- params[1]      # drift
    sigma <- params[2]   # diffusion volatility
    lambda <- params[3]  # jump intensity
    muj <- params[4]     # jump mean
    sigmaj <- params[5]  # jump volatility
    
    if (sigma <= 0 || lambda <= 0 || sigmaj <= 0) return(1e10)
    
    n <- length(ret_data)
    dt <- 1/252  # Daily data
    
    # Probability density for each observation
    log_likelihood <- 0
    
    for (i in 1:n) {
      ret <- ret_data[i]
      
      # Sum over possible number of jumps (truncate at reasonable limit)
      prob_sum <- 0
      for (k in 0:10) {
        # Poisson probability of k jumps
        poisson_prob <- exp(-lambda * dt) * (lambda * dt)^k / factorial(k)
        
        # Conditional normal density given k jumps
        if (k == 0) {
          normal_mean <- mu * dt
          normal_var <- sigma^2 * dt
        } else {
          normal_mean <- mu * dt + k * muj
          normal_var <- sigma^2 * dt + k * sigmaj^2
        }
        
        if (normal_var > 0) {
          normal_density <- dnorm(ret, normal_mean, sqrt(normal_var))
          prob_sum <- prob_sum + poisson_prob * normal_density
        }
      }
      
      if (prob_sum > 0) {
        log_likelihood <- log_likelihood + log(prob_sum)
      } else {
        return(1e10)
      }
    }
    
    return(-log_likelihood)
  }
  
  # Initial parameter estimates
  ret_mean <- mean(ret_data)
  ret_var <- var(ret_data)
  
  initial_params <- c(
    mu = ret_mean * 252,          # annualized
    sigma = sqrt(ret_var * 252) * 0.8,  # slightly lower than sample vol
    lambda = 50,                  # 50 jumps per year
    muj = -0.01,                  # negative jump mean
    sigmaj = 0.03                 # jump volatility
  )
  
  # Optimize
  tryCatch({
    mle_result <- optim(initial_params, merton_likelihood, method = "L-BFGS-B",
                       lower = c(-1, 0.001, 0.001, -0.5, 0.001),
                       upper = c(1, 2, 500, 0.5, 0.5))
    
    # Extract parameters
    estimated_params <- mle_result$par
    names(estimated_params) <- c("mu", "sigma", "lambda", "muj", "sigmaj")
    
    # Jump detection (simple threshold method)
    # Standardized returns
    std_returns <- (ret_data - estimated_params["mu"]/252) / estimated_params["sigma"]
    jump_threshold <- 3  # 3 standard deviations
    
    potential_jumps <- which(abs(std_returns) > jump_threshold)
    jump_dates <- index(returns)[potential_jumps]
    jump_magnitudes <- ret_data[potential_jumps]
    
    results <- list(
      parameters = estimated_params,
      log_likelihood = -mle_result$value,
      aic = 2 * 5 - 2 * (-mle_result$value),
      convergence = mle_result$convergence,
      jump_detection = list(
        jump_dates = jump_dates,
        jump_magnitudes = jump_magnitudes,
        n_jumps = length(potential_jumps)
      ),
      annualized_metrics = list(
        drift = estimated_params["mu"],
        diffusion_vol = estimated_params["sigma"],
        jump_intensity = estimated_params["lambda"],
        avg_jump_size = estimated_params["muj"],
        jump_volatility = estimated_params["sigmaj"]
      )
    )
    
    cat("Jump diffusion model fitted. Estimated jump intensity:", 
        round(estimated_params["lambda"], 2), "per year\n")
    
    return(results)
  }, error = function(e) {
    cat("Error fitting jump diffusion model:", e$message, "\n")
    return(NULL)
  })
}

#' Model Ensemble and Combination
#' 
#' Combines multiple models for robust predictions
#' @param returns xts object of returns
#' @param models character vector of models to include
#' @param combination_method character: "equal_weight", "aic_weight", "bma"
#' @return ensemble model results
ensemble_modeling <- function(returns, 
                             models = c("garch", "hmm", "factor"),
                             combination_method = "aic_weight") {
  
  cat("Building ensemble model with:", paste(models, collapse = ", "), "\n")
  
  model_results <- list()
  model_weights <- numeric()
  
  # Fit individual models
  if ("garch" %in% models) {
    model_results$garch <- advanced_garch_modeling(returns, "eGARCH")
    model_weights <- c(model_weights, model_results$garch$diagnostics$aic)
  }
  
  if ("hmm" %in% models) {
    model_results$hmm <- regime_switching_model(returns)
    model_weights <- c(model_weights, model_results$hmm$diagnostics$aic)
  }
  
  if ("jump_diffusion" %in% models) {
    jd_result <- jump_diffusion_model(returns)
    if (!is.null(jd_result)) {
      model_results$jump_diffusion <- jd_result
      model_weights <- c(model_weights, jd_result$aic)
    }
  }
  
  # Calculate combination weights
  if (combination_method == "aic_weight") {
    # AIC weights (lower AIC gets higher weight)
    aic_weights <- exp(-0.5 * (model_weights - min(model_weights)))
    aic_weights <- aic_weights / sum(aic_weights)
    names(aic_weights) <- names(model_results)
  } else {
    # Equal weights
    aic_weights <- rep(1/length(model_results), length(model_results))
    names(aic_weights) <- names(model_results)
  }
  
  # Ensemble volatility forecast (if GARCH models available)
  ensemble_vol_forecast <- NULL
  if ("garch" %in% names(model_results)) {
    ensemble_vol_forecast <- model_results$garch$risk_metrics$current_volatility
  }
  
  results <- list(
    individual_models = model_results,
    model_weights = aic_weights,
    ensemble_forecast = list(
      volatility = ensemble_vol_forecast
    ),
    model_comparison = data.frame(
      model = names(model_results),
      aic = model_weights,
      weight = aic_weights
    )
  )
  
  cat("Ensemble model built. Best model:", 
      names(aic_weights)[which.max(aic_weights)], "\n")
  
  return(results)
}

# Export main functions
cat("Statistical Models Module Loaded Successfully\n")
cat("Available functions:\n")
cat("- advanced_garch_modeling()\n")
cat("- multivariate_garch_dcc()\n")
cat("- regime_switching_model()\n")
cat("- factor_model_analysis()\n")
cat("- jump_diffusion_model()\n")
cat("- ensemble_modeling()\n")

# Cleanup function
cleanup_statistical_models <- function() {
  if (exists("cl") && !is.null(cl)) {
    stopCluster(cl)
    cat("Parallel cluster stopped\n")
  }
}

# Register cleanup function to run on exit
reg.finalizer(environment(), function(e) cleanup_statistical_models(), onexit = TRUE)