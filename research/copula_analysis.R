# QUANTUM-FORGE Copula Analysis
# Advanced dependence modeling and portfolio risk management
# Author: QUANTUM-FORGE Research Team
# Last Updated: 2024

# Load required libraries
suppressPackageStartupMessages({
  library(copula)
  library(VineCopula)
  library(CDVineCopulaConditional)
  library(xts)
  library(zoo)
  library(MASS)
  library(mvtnorm)
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(corrplot)
  library(RColorBrewer)
  library(parallel)
  library(doParallel)
  library(foreach)
})

cat("QUANTUM-FORGE Copula Analysis Module Initialized\n")

#' Marginal Distribution Fitting
#' 
#' Fits various distributions to marginal returns
#' @param returns numeric vector of returns
#' @param distributions character vector of distributions to test
#' @return list of fitted distributions with selection criteria
fit_marginal_distributions <- function(returns, 
                                     distributions = c("norm", "std", "sstd", "ged", "jsu")) {
  
  cat("Fitting marginal distributions for", length(returns), "observations\n")
  
  # Remove NA values
  clean_returns <- returns[!is.na(returns)]
  
  # Fit distributions
  fitted_distributions <- list()
  
  # Normal distribution
  if ("norm" %in% distributions) {
    norm_params <- list(mean = mean(clean_returns), sd = sd(clean_returns))
    fitted_distributions$norm <- list(
      params = norm_params,
      loglik = sum(dnorm(clean_returns, norm_params$mean, norm_params$sd, log = TRUE)),
      aic = -2 * sum(dnorm(clean_returns, norm_params$mean, norm_params$sd, log = TRUE)) + 2 * 2,
      ks_test = ks.test(clean_returns, "pnorm", norm_params$mean, norm_params$sd)
    )
  }
  
  # Student-t distribution
  if ("std" %in% distributions) {
    library(metRology)
    tryCatch({
      t_fit <- fitdistr(clean_returns, "t")
      fitted_distributions$std <- list(
        params = list(df = t_fit$estimate["df"], 
                     location = t_fit$estimate["m"], 
                     scale = t_fit$estimate["s"]),
        loglik = t_fit$loglik,
        aic = -2 * t_fit$loglik + 2 * length(t_fit$estimate),
        convergence = t_fit$convergence
      )
    }, error = function(e) {
      cat("Warning: Student-t fitting failed\n")
    })
  }
  
  # Generalized Error Distribution
  if ("ged" %in% distributions) {
    library(fGarch)
    tryCatch({
      ged_params <- gedFit(clean_returns)
      fitted_distributions$ged <- list(
        params = list(mean = ged_params$par[1], 
                     sd = ged_params$par[2], 
                     nu = ged_params$par[3]),
        loglik = -ged_params$minimum,
        aic = 2 * ged_params$minimum + 2 * length(ged_params$par)
      )
    }, error = function(e) {
      cat("Warning: GED fitting failed\n")
    })
  }
  
  # Select best distribution by AIC
  aic_values <- sapply(fitted_distributions, function(x) x$aic)
  best_dist <- names(aic_values)[which.min(aic_values)]
  
  # Convert to uniform margins using best distribution
  if (best_dist == "norm") {
    uniform_data <- pnorm(clean_returns, 
                         fitted_distributions$norm$params$mean,
                         fitted_distributions$norm$params$sd)
  } else if (best_dist == "std" && !is.null(fitted_distributions$std)) {
    uniform_data <- pt((clean_returns - fitted_distributions$std$params$location) / 
                      fitted_distributions$std$params$scale,
                      df = fitted_distributions$std$params$df)
  } else {
    # Fallback to empirical CDF
    uniform_data <- rank(clean_returns) / (length(clean_returns) + 1)
  }
  
  results <- list(
    fitted_distributions = fitted_distributions,
    best_distribution = best_dist,
    uniform_data = uniform_data,
    original_data = clean_returns,
    selection_criteria = data.frame(
      distribution = names(aic_values),
      aic = aic_values,
      rank = rank(aic_values)
    )
  )
  
  cat("Best marginal distribution:", best_dist, "with AIC:", round(min(aic_values), 2), "\n")
  return(results)
}

#' Bivariate Copula Analysis
#' 
#' Comprehensive analysis of bivariate copula relationships
#' @param data1 numeric vector or uniform margins
#' @param data2 numeric vector or uniform margins
#' @param copula_families character vector of copula families to test
#' @return detailed copula analysis results
bivariate_copula_analysis <- function(data1, data2, 
                                    copula_families = c("gaussian", "t", "clayton", 
                                                       "gumbel", "frank", "joe")) {
  
  cat("Performing bivariate copula analysis with", length(copula_families), "families\n")
  
  # Ensure data is on uniform margins
  if (max(data1, na.rm = TRUE) > 1 || min(data1, na.rm = TRUE) < 0) {
    u1 <- pobs(data1, ties.method = "average")
  } else {
    u1 <- data1
  }
  
  if (max(data2, na.rm = TRUE) > 1 || min(data2, na.rm = TRUE) < 0) {
    u2 <- pobs(data2, ties.method = "average")
  } else {
    u2 <- data2
  }
  
  # Remove NA pairs
  valid_idx <- !is.na(u1) & !is.na(u2)
  u1 <- u1[valid_idx]
  u2 <- u2[valid_idx]
  
  uniform_data <- cbind(u1, u2)
  
  # Fit different copula families
  copula_fits <- list()
  
  # Gaussian copula
  if ("gaussian" %in% copula_families) {
    tryCatch({
      norm_cop <- normalCopula(dim = 2)
      norm_fit <- fitCopula(norm_cop, uniform_data, method = "ml")
      copula_fits$gaussian <- list(
        copula = norm_fit@copula,
        estimate = norm_fit@estimate,
        loglik = norm_fit@loglik,
        aic = -2 * norm_fit@loglik + 2 * length(norm_fit@estimate),
        gof_test = gofCopula(norm_cop, uniform_data, N = 100)
      )
    }, error = function(e) {
      cat("Warning: Gaussian copula fitting failed\n")
    })
  }
  
  # t-Copula
  if ("t" %in% copula_families) {
    tryCatch({
      t_cop <- tCopula(dim = 2)
      t_fit <- fitCopula(t_cop, uniform_data, method = "ml")
      copula_fits$t <- list(
        copula = t_fit@copula,
        estimate = t_fit@estimate,
        loglik = t_fit@loglik,
        aic = -2 * t_fit@loglik + 2 * length(t_fit@estimate),
        gof_test = gofCopula(t_cop, uniform_data, N = 100)
      )
    }, error = function(e) {
      cat("Warning: t-copula fitting failed\n")
    })
  }
  
  # Clayton copula
  if ("clayton" %in% copula_families) {
    tryCatch({
      clayton_cop <- claytonCopula()
      clayton_fit <- fitCopula(clayton_cop, uniform_data, method = "ml")
      copula_fits$clayton <- list(
        copula = clayton_fit@copula,
        estimate = clayton_fit@estimate,
        loglik = clayton_fit@loglik,
        aic = -2 * clayton_fit@loglik + 2 * length(clayton_fit@estimate),
        gof_test = gofCopula(clayton_cop, uniform_data, N = 100)
      )
    }, error = function(e) {
      cat("Warning: Clayton copula fitting failed\n")
    })
  }
  
  # Gumbel copula
  if ("gumbel" %in% copula_families) {
    tryCatch({
      gumbel_cop <- gumbelCopula()
      gumbel_fit <- fitCopula(gumbel_cop, uniform_data, method = "ml")
      copula_fits$gumbel <- list(
        copula = gumbel_fit@copula,
        estimate = gumbel_fit@estimate,
        loglik = gumbel_fit@loglik,
        aic = -2 * gumbel_fit@loglik + 2 * length(gumbel_fit@estimate),
        gof_test = gofCopula(gumbel_cop, uniform_data, N = 100)
      )
    }, error = function(e) {
      cat("Warning: Gumbel copula fitting failed\n")
    })
  }
  
  # Frank copula
  if ("frank" %in% copula_families) {
    tryCatch({
      frank_cop <- frankCopula()
      frank_fit <- fitCopula(frank_cop, uniform_data, method = "ml")
      copula_fits$frank <- list(
        copula = frank_fit@copula,
        estimate = frank_fit@estimate,
        loglik = frank_fit@loglik,
        aic = -2 * frank_fit@loglik + 2 * length(frank_fit@estimate),
        gof_test = gofCopula(frank_cop, uniform_data, N = 100)
      )
    }, error = function(e) {
      cat("Warning: Frank copula fitting failed\n")
    })
  }
  
  # Select best copula by AIC
  aic_values <- sapply(copula_fits, function(x) x$aic)
  best_copula <- names(aic_values)[which.min(aic_values)]
  
  # Dependence measures for best copula
  best_fit <- copula_fits[[best_copula]]
  
  # Kendall's tau
  kendall_tau <- cor(u1, u2, method = "kendall")
  
  # Spearman's rho
  spearman_rho <- cor(u1, u2, method = "spearman")
  
  # Tail dependence coefficients
  if (best_copula %in% c("clayton", "gumbel", "t")) {
    if (best_copula == "clayton") {
      lambda_L <- 2^(-1/best_fit$estimate)  # Lower tail dependence
      lambda_U <- 0  # Upper tail dependence
    } else if (best_copula == "gumbel") {
      lambda_L <- 0
      lambda_U <- 2 - 2^(1/best_fit$estimate)
    } else if (best_copula == "t") {
      # For t-copula, both tails have same dependence
      rho <- best_fit$estimate[1]
      nu <- best_fit$estimate[2]
      lambda_L <- lambda_U <- 2 * pt(-sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    }
  } else {
    lambda_L <- lambda_U <- 0
  }
  
  # Conditional distributions
  conditional_quantiles <- list()
  quantile_levels <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  
  for (q in quantile_levels) {
    # P(U2 <= u2 | U1 = q)
    conditional_quantiles[[paste0("q_", q)]] <- sapply(seq(0.01, 0.99, 0.01), function(u2) {
      tryCatch({
        dCopula(cbind(q, u2), best_fit$copula) / dCopula(cbind(q, 0.5), best_fit$copula)
      }, error = function(e) NA)
    })
  }
  
  results <- list(
    copula_fits = copula_fits,
    best_copula = best_copula,
    best_fit = best_fit,
    selection_criteria = data.frame(
      copula = names(aic_values),
      aic = aic_values,
      rank = rank(aic_values)
    ),
    dependence_measures = list(
      kendall_tau = kendall_tau,
      spearman_rho = spearman_rho,
      lambda_L = lambda_L,
      lambda_U = lambda_U
    ),
    conditional_quantiles = conditional_quantiles,
    uniform_data = uniform_data
  )
  
  cat("Best bivariate copula:", best_copula, "with AIC:", round(min(aic_values), 2), "\n")
  cat("Kendall's tau:", round(kendall_tau, 3), "\n")
  
  return(results)
}

#' Vine Copula Modeling
#' 
#' High-dimensional vine copula construction and analysis
#' @param data matrix of observations (uniform margins)
#' @param vine_type character: "CVine", "DVine", "RVine"
#' @param selection_criterion character: "AIC", "BIC", "loglik"
#' @return vine copula model results
vine_copula_modeling <- function(data, vine_type = "RVine", selection_criterion = "AIC") {
  
  cat("Fitting", vine_type, "copula for", ncol(data), "variables\n")
  
  # Ensure data is on uniform margins
  if (max(data, na.rm = TRUE) > 1 || min(data, na.rm = TRUE) < 0) {
    uniform_data <- pobs(data, ties.method = "average")
  } else {
    uniform_data <- data
  }
  
  # Remove rows with NA values
  complete_cases <- complete.cases(uniform_data)
  uniform_data <- uniform_data[complete_cases, ]
  
  n_vars <- ncol(uniform_data)
  n_obs <- nrow(uniform_data)
  
  cat("Using", n_obs, "complete observations for", n_vars, "variables\n")
  
  # Fit vine copula
  tryCatch({
    if (vine_type == "RVine") {
      # R-Vine copula (most flexible)
      vine_fit <- RVineSeqEst(uniform_data, method = "mle", 
                             selectioncrit = selection_criterion,
                             familyset = c(1, 2, 3, 4, 5, 6, 13, 14, 16, 23, 24, 26, 33, 34, 36))
      
      vine_structure <- vine_fit$RVM
      
    } else if (vine_type == "CVine") {
      # C-Vine copula
      vine_fit <- RVineSeqEst(uniform_data, method = "mle",
                             selectioncrit = selection_criterion,
                             type = 1)  # C-Vine
      vine_structure <- vine_fit$RVM
      
    } else if (vine_type == "DVine") {
      # D-Vine copula
      vine_fit <- RVineSeqEst(uniform_data, method = "mle",
                             selectioncrit = selection_criterion,
                             type = 2)  # D-Vine
      vine_structure <- vine_fit$RVM
    }
    
    # Model diagnostics
    vine_loglik <- RVineLogLik(uniform_data, vine_fit$RVM)
    n_params <- sum(vine_fit$RVM$par != 0) + sum(vine_fit$RVM$par2 != 0)
    vine_aic <- -2 * vine_loglik + 2 * n_params
    vine_bic <- -2 * vine_loglik + log(n_obs) * n_params
    
    # Goodness of fit tests
    gof_test <- RVineGofTest(uniform_data, vine_fit$RVM, method = "White")
    
    # Simulate from fitted vine
    vine_sim <- RVineSim(n_obs, vine_fit$RVM)
    
    # Calculate pairwise correlations
    emp_corr <- cor(uniform_data, method = "spearman")
    sim_corr <- cor(vine_sim, method = "spearman")
    corr_diff <- abs(emp_corr - sim_corr)
    
    # Conditional distributions
    # Example: P(X3 | X1, X2) for first few observations
    conditional_analysis <- list()
    if (n_vars >= 3) {
      for (i in 1:min(10, n_obs)) {
        conditioning_vars <- uniform_data[i, 1:(n_vars-1)]
        conditional_analysis[[i]] <- RVineCondDistr(uniform_data[i, n_vars], 
                                                   conditioning_vars, 
                                                   vine_fit$RVM)
      }
    }
    
    # Risk measures
    # Portfolio VaR using vine copula simulation
    portfolio_weights <- rep(1/n_vars, n_vars)  # Equal weights
    
    # Simulate portfolio returns (assuming standard normal margins for simplicity)
    portfolio_sim <- apply(vine_sim, 1, function(row) {
      sum(portfolio_weights * qnorm(row))
    })
    
    var_95 <- quantile(portfolio_sim, 0.05)
    var_99 <- quantile(portfolio_sim, 0.01)
    es_95 <- mean(portfolio_sim[portfolio_sim <= var_95])
    es_99 <- mean(portfolio_sim[portfolio_sim <= var_99])
    
    results <- list(
      vine_model = vine_fit,
      vine_structure = vine_structure,
      vine_type = vine_type,
      diagnostics = list(
        loglik = vine_loglik,
        aic = vine_aic,
        bic = vine_bic,
        n_parameters = n_params,
        gof_test = gof_test
      ),
      simulated_data = vine_sim,
      correlation_comparison = list(
        empirical = emp_corr,
        simulated = sim_corr,
        difference = corr_diff,
        max_diff = max(corr_diff[upper.tri(corr_diff)])
      ),
      conditional_analysis = conditional_analysis,
      risk_measures = list(
        var_95 = var_95,
        var_99 = var_99,
        es_95 = es_95,
        es_99 = es_99
      ),
      uniform_data = uniform_data
    )
    
    cat("Vine copula fitted successfully\n")
    cat("Log-likelihood:", round(vine_loglik, 2), "\n")
    cat("AIC:", round(vine_aic, 2), "\n")
    cat("Max correlation difference:", round(max(corr_diff[upper.tri(corr_diff)]), 4), "\n")
    
    return(results)
    
  }, error = function(e) {
    cat("Error fitting vine copula:", e$message, "\n")
    return(NULL)
  })
}

#' Copula-based Portfolio Risk Management
#' 
#' Portfolio risk analysis using copula models
#' @param returns_data matrix of asset returns
#' @param portfolio_weights numeric vector of portfolio weights
#' @param confidence_levels numeric vector of confidence levels for VaR
#' @param simulation_size integer: number of Monte Carlo simulations
#' @return comprehensive portfolio risk analysis
copula_portfolio_risk <- function(returns_data, 
                                 portfolio_weights = NULL,
                                 confidence_levels = c(0.95, 0.99),
                                 simulation_size = 10000) {
  
  cat("Performing copula-based portfolio risk analysis\n")
  
  # Default equal weights
  if (is.null(portfolio_weights)) {
    portfolio_weights <- rep(1/ncol(returns_data), ncol(returns_data))
  }
  
  n_assets <- ncol(returns_data)
  asset_names <- colnames(returns_data)
  if (is.null(asset_names)) {
    asset_names <- paste0("Asset_", 1:n_assets)
  }
  
  # Step 1: Fit marginal distributions
  marginal_fits <- list()
  uniform_margins <- matrix(NA, nrow = nrow(returns_data), ncol = n_assets)
  
  for (i in 1:n_assets) {
    cat("Fitting marginal distribution for", asset_names[i], "\n")
    marginal_fits[[asset_names[i]]] <- fit_marginal_distributions(returns_data[, i])
    uniform_margins[, i] <- marginal_fits[[asset_names[i]]]$uniform_data
  }
  
  # Step 2: Fit vine copula to uniform margins
  vine_model <- vine_copula_modeling(uniform_margins, vine_type = "RVine")
  
  if (is.null(vine_model)) {
    cat("Vine copula fitting failed, using normal copula\n")
    # Fallback to multivariate normal
    emp_corr <- cor(uniform_margins, use = "complete.obs", method = "spearman")
    norm_cop <- normalCopula(P2p(emp_corr), dim = n_assets, dispstr = "un")
    
    # Simple simulation from normal copula
    vine_simulation <- rCopula(simulation_size, norm_cop)
  } else {
    # Simulate from vine copula
    vine_simulation <- RVineSim(simulation_size, vine_model$vine_model$RVM)
  }
  
  # Step 3: Transform back to original margins and calculate portfolio returns
  portfolio_simulations <- numeric(simulation_size)
  
  for (i in 1:simulation_size) {
    # Transform uniform margins back to return distributions
    simulated_returns <- numeric(n_assets)
    
    for (j in 1:n_assets) {
      u_sim <- vine_simulation[i, j]
      best_dist <- marginal_fits[[asset_names[j]]]$best_distribution
      
      if (best_dist == "norm") {
        params <- marginal_fits[[asset_names[j]]]$fitted_distributions$norm$params
        simulated_returns[j] <- qnorm(u_sim, params$mean, params$sd)
      } else if (best_dist == "std" && 
                 !is.null(marginal_fits[[asset_names[j]]]$fitted_distributions$std)) {
        params <- marginal_fits[[asset_names[j]]]$fitted_distributions$std$params
        simulated_returns[j] <- params$location + params$scale * 
          qt(u_sim, df = params$df)
      } else {
        # Fallback to empirical quantile
        original_data <- marginal_fits[[asset_names[j]]]$original_data
        simulated_returns[j] <- quantile(original_data, u_sim, na.rm = TRUE)
      }
    }
    
    # Calculate portfolio return
    portfolio_simulations[i] <- sum(portfolio_weights * simulated_returns)
  }
  
  # Step 4: Calculate risk measures
  risk_measures <- list()
  
  for (alpha in confidence_levels) {
    var_level <- quantile(portfolio_simulations, 1 - alpha)
    es_level <- mean(portfolio_simulations[portfolio_simulations <= var_level])
    
    risk_measures[[paste0("VaR_", alpha)]] <- var_level
    risk_measures[[paste0("ES_", alpha)]] <- es_level
  }
  
  # Component VaR (marginal contribution to VaR)
  component_var <- numeric(n_assets)
  epsilon <- 0.001
  
  for (i in 1:n_assets) {
    # Perturb weight slightly
    perturbed_weights <- portfolio_weights
    perturbed_weights[i] <- perturbed_weights[i] + epsilon
    perturbed_weights <- perturbed_weights / sum(perturbed_weights)  # Renormalize
    
    # Recalculate portfolio VaR
    perturbed_portfolio <- apply(vine_simulation, 1, function(row) {
      simulated_returns <- numeric(n_assets)
      for (j in 1:n_assets) {
        u_sim <- row[j]
        best_dist <- marginal_fits[[asset_names[j]]]$best_distribution
        
        if (best_dist == "norm") {
          params <- marginal_fits[[asset_names[j]]]$fitted_distributions$norm$params
          simulated_returns[j] <- qnorm(u_sim, params$mean, params$sd)
        } else {
          original_data <- marginal_fits[[asset_names[j]]]$original_data
          simulated_returns[j] <- quantile(original_data, u_sim, na.rm = TRUE)
        }
      }
      sum(perturbed_weights * simulated_returns)
    })
    
    perturbed_var <- quantile(perturbed_portfolio, 0.05)  # 95% VaR
    component_var[i] <- (perturbed_var - risk_measures$VaR_0.95) / epsilon
  }
  
  names(component_var) <- asset_names
  
  # Diversification ratio
  individual_vars <- sapply(1:n_assets, function(i) {
    individual_portfolio <- numeric(simulation_size)
    for (j in 1:simulation_size) {
      u_sim <- vine_simulation[j, i]
      best_dist <- marginal_fits[[asset_names[i]]]$best_distribution
      
      if (best_dist == "norm") {
        params <- marginal_fits[[asset_names[i]]]$fitted_distributions$norm$params
        individual_portfolio[j] <- qnorm(u_sim, params$mean, params$sd)
      } else {
        original_data <- marginal_fits[[asset_names[i]]]$original_data
        individual_portfolio[j] <- quantile(original_data, u_sim, na.rm = TRUE)
      }
    }
    quantile(individual_portfolio, 0.05)
  })
  
  weighted_individual_var <- sum(portfolio_weights * individual_vars)
  diversification_ratio <- weighted_individual_var / risk_measures$VaR_0.95
  
  results <- list(
    marginal_fits = marginal_fits,
    vine_model = vine_model,
    portfolio_simulations = portfolio_simulations,
    risk_measures = risk_measures,
    component_var = component_var,
    diversification_ratio = diversification_ratio,
    portfolio_weights = portfolio_weights,
    simulation_summary = list(
      mean = mean(portfolio_simulations),
      sd = sd(portfolio_simulations),
      skewness = moments::skewness(portfolio_simulations),
      kurtosis = moments::kurtosis(portfolio_simulations),
      min = min(portfolio_simulations),
      max = max(portfolio_simulations)
    )
  )
  
  cat("Portfolio risk analysis completed\n")
  cat("Portfolio VaR (95%):", round(risk_measures$VaR_0.95 * 100, 2), "%\n")
  cat("Portfolio ES (95%):", round(risk_measures$ES_0.95 * 100, 2), "%\n")
  cat("Diversification ratio:", round(diversification_ratio, 2), "\n")
  
  return(results)
}

#' Copula Backtesting
#' 
#' Backtest copula models for risk management validation
#' @param historical_returns matrix of historical returns
#' @param portfolio_weights numeric vector of weights
#' @param window_size integer: rolling window size for model estimation
#' @param confidence_level numeric: confidence level for VaR
#' @return backtesting results with violation statistics
copula_backtest <- function(historical_returns, 
                           portfolio_weights,
                           window_size = 252,
                           confidence_level = 0.05) {
  
  cat("Starting copula model backtesting\n")
  cat("Window size:", window_size, "days\n")
  
  n_obs <- nrow(historical_returns)
  n_assets <- ncol(historical_returns)
  
  # Calculate actual portfolio returns
  actual_portfolio_returns <- as.numeric(historical_returns %*% portfolio_weights)
  
  # Rolling window backtest
  var_forecasts <- numeric(n_obs - window_size)
  es_forecasts <- numeric(n_obs - window_size)
  backtest_dates <- (window_size + 1):n_obs
  
  cat("Running", length(backtest_dates), "backtesting iterations\n")
  
  for (i in seq_along(backtest_dates)) {
    if (i %% 50 == 0) cat("Processed", i, "out of", length(backtest_dates), "iterations\n")
    
    # Extract training window
    train_end <- backtest_dates[i] - 1
    train_start <- train_end - window_size + 1
    train_data <- historical_returns[train_start:train_end, ]
    
    # Fit copula model on training data
    tryCatch({
      copula_risk <- copula_portfolio_risk(train_data, 
                                          portfolio_weights, 
                                          confidence_levels = confidence_level,
                                          simulation_size = 5000)
      
      var_forecasts[i] <- copula_risk$risk_measures[[paste0("VaR_", confidence_level)]]
      es_forecasts[i] <- copula_risk$risk_measures[[paste0("ES_", confidence_level)]]
      
    }, error = function(e) {
      # Fallback to historical simulation
      train_portfolio <- as.numeric(train_data %*% portfolio_weights)
      var_forecasts[i] <<- quantile(train_portfolio, confidence_level)
      es_forecasts[i] <<- mean(train_portfolio[train_portfolio <= var_forecasts[i]])
    })
  }
  
  # Calculate actual returns for backtest period
  actual_returns_backtest <- actual_portfolio_returns[backtest_dates]
  
  # VaR violations
  var_violations <- actual_returns_backtest < var_forecasts
  violation_rate <- mean(var_violations)
  expected_violation_rate <- confidence_level
  
  # Unconditional coverage test (Kupiec test)
  n_violations <- sum(var_violations)
  n_observations <- length(var_violations)
  
  kupiec_lr <- -2 * log(
    (confidence_level^n_violations * (1 - confidence_level)^(n_observations - n_violations)) /
    ((n_violations/n_observations)^n_violations * 
     (1 - n_violations/n_observations)^(n_observations - n_violations))
  )
  kupiec_pvalue <- 1 - pchisq(kupiec_lr, df = 1)
  
  # Conditional coverage test (Christoffersen test)
  # Transition matrix for violations
  violation_transitions <- table(var_violations[-1], var_violations[-length(var_violations)])
  
  if (nrow(violation_transitions) == 2 && ncol(violation_transitions) == 2) {
    n00 <- violation_transitions[1, 1]  # No violation followed by no violation
    n01 <- violation_transitions[1, 2]  # No violation followed by violation
    n10 <- violation_transitions[2, 1]  # Violation followed by no violation
    n11 <- violation_transitions[2, 2]  # Violation followed by violation
    
    # Independence test
    pi01 <- n01 / (n00 + n01)
    pi11 <- n11 / (n10 + n11)
    pi <- (n01 + n11) / (n00 + n01 + n10 + n11)
    
    if (pi01 > 0 && pi11 > 0 && pi > 0) {
      christoffersen_lr <- -2 * log(
        (pi^(n01 + n11) * (1 - pi)^(n00 + n10)) /
        (pi01^n01 * (1 - pi01)^n00 * pi11^n11 * (1 - pi11)^n10)
      )
      christoffersen_pvalue <- 1 - pchisq(christoffersen_lr, df = 1)
    } else {
      christoffersen_lr <- NA
      christoffersen_pvalue <- NA
    }
  } else {
    christoffersen_lr <- NA
    christoffersen_pvalue <- NA
  }
  
  # Expected Shortfall backtesting (using ES violations)
  es_violations <- actual_returns_backtest[var_violations]
  es_forecasts_violations <- es_forecasts[var_violations]
  
  if (length(es_violations) > 0) {
    es_relative_violations <- es_violations < es_forecasts_violations
    es_violation_rate <- mean(es_relative_violations)
  } else {
    es_violation_rate <- 0
  }
  
  # Performance metrics
  mean_absolute_error <- mean(abs(actual_returns_backtest - var_forecasts))
  root_mean_squared_error <- sqrt(mean((actual_returns_backtest - var_forecasts)^2))
  
  results <- list(
    var_forecasts = var_forecasts,
    es_forecasts = es_forecasts,
    actual_returns = actual_returns_backtest,
    var_violations = var_violations,
    violation_statistics = list(
      violation_rate = violation_rate,
      expected_violation_rate = expected_violation_rate,
      n_violations = n_violations,
      n_observations = n_observations
    ),
    statistical_tests = list(
      kupiec_lr = kupiec_lr,
      kupiec_pvalue = kupiec_pvalue,
      christoffersen_lr = christoffersen_lr,
      christoffersen_pvalue = christoffersen_pvalue
    ),
    es_validation = list(
      es_violation_rate = es_violation_rate,
      n_es_violations = length(es_violations)
    ),
    performance_metrics = list(
      mae = mean_absolute_error,
      rmse = root_mean_squared_error
    ),
    backtest_dates = backtest_dates
  )
  
  cat("Backtesting completed\n")
  cat("VaR violation rate:", round(violation_rate * 100, 2), "% (expected:", 
      round(expected_violation_rate * 100, 2), "%)\n")
  cat("Kupiec test p-value:", round(kupiec_pvalue, 4), "\n")
  
  return(results)
}

# Export main functions
cat("Copula Analysis Module Loaded Successfully\n")
cat("Available functions:\n")
cat("- fit_marginal_distributions()\n")
cat("- bivariate_copula_analysis()\n")
cat("- vine_copula_modeling()\n")
cat("- copula_portfolio_risk()\n")
cat("- copula_backtest()\n")