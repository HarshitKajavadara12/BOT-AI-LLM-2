# QUANTUM-FORGE Extreme Value Theory
# Advanced tail risk modeling and extreme events analysis
# Author: QUANTUM-FORGE Research Team
# Last Updated: 2024

# Load required libraries
suppressPackageStartupMessages({
  library(evd)
  library(extRemes)
  library(eva)
  library(POT)
  library(fExtremes)
  library(QRM)
  library(ismev)
  library(xts)
  library(zoo)
  library(quantmod)
  library(PerformanceAnalytics)
  library(VaR)
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  library(RColorBrewer)
  library(parallel)
  library(doParallel)
  library(foreach)
})

cat("QUANTUM-FORGE Extreme Value Theory Module Initialized\n")

#' Block Maxima Approach
#' 
#' Fits Generalized Extreme Value (GEV) distribution to block maxima
#' @param data numeric vector of observations
#' @param block_size integer: size of each block (default: 22 for monthly maxima)
#' @param method character: estimation method ("MLE", "PWM", "Bayesian")
#' @return GEV model results with diagnostics
block_maxima_analysis <- function(data, block_size = 22, method = "MLE") {
  
  cat("Performing Block Maxima analysis with block size:", block_size, "\n")
  
  # Remove NA values
  clean_data <- data[!is.na(data)]
  n_obs <- length(clean_data)
  
  # Create blocks and extract maxima
  n_blocks <- floor(n_obs / block_size)
  block_maxima <- numeric(n_blocks)
  
  for (i in 1:n_blocks) {
    start_idx <- (i - 1) * block_size + 1
    end_idx <- i * block_size
    block_maxima[i] <- max(clean_data[start_idx:end_idx])
  }
  
  cat("Extracted", n_blocks, "block maxima from", n_obs, "observations\n")
  
  # Fit GEV distribution
  if (method == "MLE") {
    tryCatch({
      gev_fit <- fevd(block_maxima, type = "GEV", method = "MLE")
      
      # Extract parameters
      location <- gev_fit$results$par[1]  # mu
      scale <- gev_fit$results$par[2]     # sigma
      shape <- gev_fit$results$par[3]     # xi
      
      # Standard errors
      se_location <- sqrt(diag(gev_fit$results$hessian.inverse))[1]
      se_scale <- sqrt(diag(gev_fit$results$hessian.inverse))[2]
      se_shape <- sqrt(diag(gev_fit$results$hessian.inverse))[3]
      
      # Log-likelihood and AIC
      loglik <- gev_fit$results$value
      aic <- -2 * loglik + 2 * 3
      bic <- -2 * loglik + log(n_blocks) * 3
      
    }, error = function(e) {
      cat("MLE fitting failed, using method of moments\n")
      # Fallback to method of moments
      mom_params <- gevFit(block_maxima, type = "gev")@fit$par.ests
      location <- mom_params[1]
      scale <- mom_params[2]
      shape <- mom_params[3]
      loglik <- NA
      aic <- NA
      bic <- NA
      se_location <- se_scale <- se_shape <- NA
    })
    
  } else if (method == "PWM") {
    # Probability Weighted Moments
    tryCatch({
      pwm_fit <- gevFit(block_maxima, type = "pwm")
      location <- pwm_fit@fit$par.ests[1]
      scale <- pwm_fit@fit$par.ests[2]
      shape <- pwm_fit@fit$par.ests[3]
      loglik <- pwm_fit@fit$llh
      aic <- -2 * loglik + 2 * 3
      bic <- -2 * loglik + log(n_blocks) * 3
      se_location <- se_scale <- se_shape <- NA
    }, error = function(e) {
      cat("PWM fitting failed\n")
      return(NULL)
    })
  }
  
  # Goodness of fit tests
  # Anderson-Darling test
  theoretical_quantiles <- qgev(ppoints(n_blocks), loc = location, scale = scale, shape = shape)
  ad_stat <- ad.test(sort(block_maxima), pgev, loc = location, scale = scale, shape = shape)
  
  # Kolmogorov-Smirnov test
  ks_stat <- ks.test(block_maxima, pgev, loc = location, scale = scale, shape = shape)
  
  # Calculate return levels
  return_periods <- c(2, 5, 10, 20, 50, 100)
  return_levels <- numeric(length(return_periods))
  return_level_se <- numeric(length(return_periods))
  
  for (i in seq_along(return_periods)) {
    T <- return_periods[i]
    p <- 1 - 1/T
    
    # Return level calculation
    if (abs(shape) < 1e-6) {
      # Gumbel case (shape ≈ 0)
      return_levels[i] <- location - scale * log(-log(p))
    } else {
      # Fréchet or Weibull case
      return_levels[i] <- location + scale * ((-log(p))^(-shape) - 1) / shape
    }
    
    # Delta method for standard errors (approximate)
    if (!is.na(se_location) && !is.na(se_scale) && !is.na(se_shape)) {
      if (abs(shape) < 1e-6) {
        # Gumbel case
        grad_mu <- 1
        grad_sigma <- -log(-log(p))
        grad_xi <- 0  # Approximation for small xi
      } else {
        # General case
        z_p <- (-log(p))^(-shape)
        grad_mu <- 1
        grad_sigma <- (z_p - 1) / shape
        grad_xi <- scale * (z_p * log(-log(p)) * shape^(-1) - (z_p - 1) * shape^(-2))
      }
      
      # Covariance matrix (diagonal approximation)
      var_rl <- grad_mu^2 * se_location^2 + grad_sigma^2 * se_scale^2 + grad_xi^2 * se_shape^2
      return_level_se[i] <- sqrt(var_rl)
    }
  }
  
  # Confidence intervals for return levels
  ci_lower <- return_levels - 1.96 * return_level_se
  ci_upper <- return_levels + 1.96 * return_level_se
  
  # Model diagnostics
  # Probability plot
  theoretical_probs <- pgev(sort(block_maxima), loc = location, scale = scale, shape = shape)
  empirical_probs <- ppoints(n_blocks)
  
  # Q-Q plot
  qq_theoretical <- qgev(ppoints(n_blocks), loc = location, scale = scale, shape = shape)
  qq_empirical <- sort(block_maxima)
  
  # Calculate effective sample size (for clustering adjustment)
  # Simple lag-1 autocorrelation check
  if (n_blocks > 3) {
    autocorr <- cor(block_maxima[-1], block_maxima[-n_blocks])
    effective_n <- n_blocks * (1 - autocorr) / (1 + autocorr)
  } else {
    autocorr <- 0
    effective_n <- n_blocks
  }
  
  results <- list(
    parameters = list(
      location = location,
      scale = scale,
      shape = shape,
      se_location = se_location,
      se_scale = se_scale,
      se_shape = se_shape
    ),
    model_fit = list(
      method = method,
      loglik = loglik,
      aic = aic,
      bic = bic,
      n_blocks = n_blocks,
      effective_sample_size = effective_n
    ),
    goodness_of_fit = list(
      anderson_darling = ad_stat,
      kolmogorov_smirnov = ks_stat,
      autocorrelation = autocorr
    ),
    return_levels = data.frame(
      return_period = return_periods,
      return_level = return_levels,
      se = return_level_se,
      ci_lower = ci_lower,
      ci_upper = ci_upper
    ),
    diagnostics = list(
      block_maxima = block_maxima,
      pp_empirical = empirical_probs,
      pp_theoretical = theoretical_probs,
      qq_empirical = qq_empirical,
      qq_theoretical = qq_theoretical
    ),
    original_data = clean_data,
    block_size = block_size
  )
  
  # Interpret shape parameter
  if (shape > 0.1) {
    tail_type <- "Heavy-tailed (Fréchet domain)"
  } else if (shape < -0.1) {
    tail_type <- "Light-tailed (Weibull domain)"
  } else {
    tail_type <- "Exponential-tailed (Gumbel domain)"
  }
  
  cat("GEV parameters: location =", round(location, 4), 
      ", scale =", round(scale, 4), 
      ", shape =", round(shape, 4), "\n")
  cat("Tail behavior:", tail_type, "\n")
  cat("100-year return level:", round(return_levels[6], 4), "\n")
  
  return(results)
}

#' Peaks Over Threshold (POT) Analysis
#' 
#' Fits Generalized Pareto Distribution to exceedances over threshold
#' @param data numeric vector of observations
#' @param threshold numeric: threshold value (if NULL, estimated automatically)
#' @param threshold_method character: method for threshold selection
#' @return GPD model results with diagnostics
peaks_over_threshold <- function(data, threshold = NULL, 
                                threshold_method = "mean_residual_life") {
  
  cat("Performing Peaks Over Threshold analysis\n")
  
  # Remove NA values
  clean_data <- data[!is.na(data)]
  n_obs <- length(clean_data)
  
  # Automatic threshold selection if not provided
  if (is.null(threshold)) {
    if (threshold_method == "mean_residual_life") {
      # Mean residual life plot method
      thresholds <- quantile(clean_data, seq(0.7, 0.95, 0.01))
      mrl_values <- sapply(thresholds, function(u) {
        excesses <- clean_data[clean_data > u] - u
        if (length(excesses) > 10) {
          mean(excesses)
        } else {
          NA
        }
      })
      
      # Find threshold where MRL is approximately linear
      # Simple approach: take 90th percentile
      threshold <- quantile(clean_data, 0.9)
      
    } else if (threshold_method == "parameter_stability") {
      # Parameter stability method
      thresholds <- quantile(clean_data, seq(0.8, 0.95, 0.01))
      shape_estimates <- numeric(length(thresholds))
      
      for (i in seq_along(thresholds)) {
        excesses <- clean_data[clean_data > thresholds[i]] - thresholds[i]
        if (length(excesses) > 20) {
          tryCatch({
            gpd_temp <- fevd(excesses, type = "GP", method = "MLE")
            shape_estimates[i] <- gpd_temp$results$par[2]
          }, error = function(e) {
            shape_estimates[i] <- NA
          })
        } else {
          shape_estimates[i] <- NA
        }
      }
      
      # Find region of stability (minimal variance)
      valid_idx <- !is.na(shape_estimates)
      if (sum(valid_idx) > 5) {
        stable_region <- which.min(sapply(5:sum(valid_idx), function(i) {
          var(shape_estimates[valid_idx][max(1, i-4):i], na.rm = TRUE)
        }))
        threshold <- thresholds[valid_idx][stable_region]
      } else {
        threshold <- quantile(clean_data, 0.9)
      }
    }
  }
  
  cat("Using threshold:", round(threshold, 4), "\n")
  
  # Extract exceedances
  exceedances <- clean_data[clean_data > threshold]
  excesses <- exceedances - threshold
  n_exceedances <- length(exceedances)
  
  if (n_exceedances < 10) {
    cat("Warning: Very few exceedances (", n_exceedances, "). Consider lowering threshold.\n")
  }
  
  cat("Number of exceedances:", n_exceedances, "(", 
      round(n_exceedances/n_obs * 100, 2), "% of data)\n")
  
  # Fit GPD to excesses
  tryCatch({
    gpd_fit <- fevd(excesses, type = "GP", method = "MLE")
    
    # Extract parameters
    scale <- gpd_fit$results$par[1]    # sigma
    shape <- gpd_fit$results$par[2]    # xi
    
    # Standard errors
    if (!is.null(gpd_fit$results$hessian.inverse)) {
      se_scale <- sqrt(diag(gpd_fit$results$hessian.inverse))[1]
      se_shape <- sqrt(diag(gpd_fit$results$hessian.inverse))[2]
    } else {
      se_scale <- se_shape <- NA
    }
    
    # Log-likelihood and information criteria
    loglik <- gpd_fit$results$value
    aic <- -2 * loglik + 2 * 2
    bic <- -2 * loglik + log(n_exceedances) * 2
    
  }, error = function(e) {
    cat("MLE fitting failed, using method of moments\n")
    # Method of moments fallback
    excess_mean <- mean(excesses)
    excess_var <- var(excesses)
    
    # Method of moments estimators
    shape <- 0.5 * (excess_mean^2 / excess_var - 1)
    scale <- 0.5 * excess_mean * (excess_mean^2 / excess_var + 1)
    
    loglik <- NA
    aic <- NA
    bic <- NA
    se_scale <- se_shape <- NA
  })
  
  # Goodness of fit tests
  theoretical_quantiles <- qgpd(ppoints(n_exceedances), loc = 0, scale = scale, shape = shape)
  
  # Anderson-Darling test for GPD
  ad_stat <- tryCatch({
    ad.test(sort(excesses), pgpd, loc = 0, scale = scale, shape = shape)
  }, error = function(e) {
    list(statistic = NA, p.value = NA)
  })
  
  # Kolmogorov-Smirnov test
  ks_stat <- ks.test(excesses, pgpd, loc = 0, scale = scale, shape = shape)
  
  # Calculate return levels for exceedances
  # First, estimate the rate of exceedances
  lambda <- n_exceedances / n_obs  # Rate of exceedances per observation
  
  return_periods <- c(2, 5, 10, 20, 50, 100)
  return_levels <- numeric(length(return_periods))
  return_level_se <- numeric(length(return_periods))
  
  for (i in seq_along(return_periods)) {
    T <- return_periods[i]
    
    # For POT, the return level is: threshold + scale * ((λT)^ξ - 1) / ξ
    if (abs(shape) < 1e-6) {
      # Exponential case (shape ≈ 0)
      return_levels[i] <- threshold + scale * log(lambda * T)
    } else {
      # General GPD case
      return_levels[i] <- threshold + scale * ((lambda * T)^shape - 1) / shape
    }
    
    # Delta method for standard errors
    if (!is.na(se_scale) && !is.na(se_shape)) {
      if (abs(shape) < 1e-6) {
        # Exponential case
        grad_sigma <- log(lambda * T)
        grad_xi <- 0
      } else {
        # General case
        z_T <- (lambda * T)^shape
        grad_sigma <- (z_T - 1) / shape
        grad_xi <- scale * (z_T * log(lambda * T) * shape^(-1) - (z_T - 1) * shape^(-2))
      }
      
      var_rl <- grad_sigma^2 * se_scale^2 + grad_xi^2 * se_shape^2
      return_level_se[i] <- sqrt(var_rl)
    }
  }
  
  # Confidence intervals
  ci_lower <- return_levels - 1.96 * return_level_se
  ci_upper <- return_levels + 1.96 * return_level_se
  
  # Calculate VaR and Expected Shortfall
  confidence_levels <- c(0.95, 0.99, 0.995, 0.999)
  var_estimates <- numeric(length(confidence_levels))
  es_estimates <- numeric(length(confidence_levels))
  
  for (i in seq_along(confidence_levels)) {
    alpha <- confidence_levels[i]
    
    # VaR calculation using POT
    if (abs(shape) < 1e-6) {
      # Exponential case
      var_estimates[i] <- threshold + scale * log((1 - alpha) / lambda)
    } else {
      # General GPD case
      var_estimates[i] <- threshold + scale * (((1 - alpha) / lambda)^(-shape) - 1) / shape
    }
    
    # Expected Shortfall calculation
    if (shape < 1) {
      es_estimates[i] <- var_estimates[i] + (scale - shape * threshold) / (1 - shape)
    } else {
      es_estimates[i] <- NA  # ES undefined for shape >= 1
    }
  }
  
  # Model diagnostics
  # Probability plot
  theoretical_probs <- pgpd(sort(excesses), loc = 0, scale = scale, shape = shape)
  empirical_probs <- ppoints(n_exceedances)
  
  # Q-Q plot
  qq_theoretical <- qgpd(ppoints(n_exceedances), loc = 0, scale = scale, shape = shape)
  qq_empirical <- sort(excesses)
  
  results <- list(
    threshold = threshold,
    parameters = list(
      scale = scale,
      shape = shape,
      se_scale = se_scale,
      se_shape = se_shape
    ),
    model_fit = list(
      loglik = loglik,
      aic = aic,
      bic = bic,
      n_exceedances = n_exceedances,
      exceedance_rate = lambda
    ),
    goodness_of_fit = list(
      anderson_darling = ad_stat,
      kolmogorov_smirnov = ks_stat
    ),
    return_levels = data.frame(
      return_period = return_periods,
      return_level = return_levels,
      se = return_level_se,
      ci_lower = ci_lower,
      ci_upper = ci_upper
    ),
    risk_measures = data.frame(
      confidence_level = confidence_levels,
      var = var_estimates,
      expected_shortfall = es_estimates
    ),
    diagnostics = list(
      excesses = excesses,
      exceedances = exceedances,
      pp_empirical = empirical_probs,
      pp_theoretical = theoretical_probs,
      qq_empirical = qq_empirical,
      qq_theoretical = qq_theoretical
    ),
    original_data = clean_data
  )
  
  # Interpret shape parameter
  if (shape > 0.1) {
    tail_type <- "Heavy-tailed (polynomial decay)"
  } else if (shape < -0.1) {
    tail_type <- "Light-tailed (finite upper bound)"
  } else {
    tail_type <- "Exponential-tailed"
  }
  
  cat("GPD parameters: scale =", round(scale, 4), 
      ", shape =", round(shape, 4), "\n")
  cat("Tail behavior:", tail_type, "\n")
  cat("99% VaR:", round(var_estimates[2], 4), "\n")
  
  return(results)
}

#' Multivariate Extreme Value Analysis
#' 
#' Analysis of joint extreme events using multivariate EVT
#' @param data matrix of multivariate observations
#' @param method character: "logistic", "husler_reiss", "tawn"
#' @param marginal_method character: method for marginal transformation
#' @return multivariate extreme value model results
multivariate_extremes <- function(data, method = "logistic", 
                                 marginal_method = "empirical") {
  
  cat("Performing multivariate extreme value analysis\n")
  
  # Remove rows with any NA values
  complete_data <- data[complete.cases(data), ]
  n_obs <- nrow(complete_data)
  n_vars <- ncol(complete_data)
  
  if (is.null(colnames(complete_data))) {
    colnames(complete_data) <- paste0("Var", 1:n_vars)
  }
  
  cat("Using", n_obs, "complete observations for", n_vars, "variables\n")
  
  # Transform margins to standard form
  if (marginal_method == "empirical") {
    # Transform to unit Fréchet margins using empirical distribution
    frechet_data <- matrix(NA, nrow = n_obs, ncol = n_vars)
    for (i in 1:n_vars) {
      # Convert to uniform margins first
      uniform_margins <- rank(complete_data[, i]) / (n_obs + 1)
      # Transform to standard Fréchet: F^(-1)(u) = -1/log(u)
      frechet_data[, i] <- -1 / log(uniform_margins)
    }
    colnames(frechet_data) <- colnames(complete_data)
    
  } else if (marginal_method == "gev") {
    # Fit GEV to each margin and transform
    frechet_data <- matrix(NA, nrow = n_obs, ncol = n_vars)
    marginal_fits <- list()
    
    for (i in 1:n_vars) {
      # Use block maxima approach for each margin
      margin_maxima <- block_maxima_analysis(complete_data[, i], block_size = 22)
      marginal_fits[[i]] <- margin_maxima
      
      # Transform using fitted GEV
      gev_params <- margin_maxima$parameters
      uniform_margins <- pgev(complete_data[, i], 
                             loc = gev_params$location,
                             scale = gev_params$scale,
                             shape = gev_params$shape)
      # Ensure values are in (0,1)
      uniform_margins <- pmax(pmin(uniform_margins, 0.999), 0.001)
      # Transform to Fréchet
      frechet_data[, i] <- -1 / log(uniform_margins)
    }
    colnames(frechet_data) <- colnames(complete_data)
  }
  
  # Fit multivariate extreme value model
  if (method == "logistic") {
    # Logistic model fitting
    cat("Fitting logistic extreme value model\n")
    
    tryCatch({
      # Use method of moments or maximum likelihood
      # Simple approach: estimate dependence parameter from data
      
      # Calculate componentwise maxima for different block sizes
      block_sizes <- c(10, 20, 30)
      dependence_params <- numeric(length(block_sizes))
      
      for (b in seq_along(block_sizes)) {
        block_size <- block_sizes[b]
        n_blocks <- floor(n_obs / block_size)
        
        if (n_blocks > 10) {
          block_max_matrix <- matrix(NA, nrow = n_blocks, ncol = n_vars)
          
          for (i in 1:n_blocks) {
            start_idx <- (i - 1) * block_size + 1
            end_idx <- i * block_size
            block_max_matrix[i, ] <- apply(frechet_data[start_idx:end_idx, ], 2, max)
          }
          
          # Estimate dependence parameter using Pickands estimator
          # For bivariate case: A(t) = max(t, 1-t) + θ * t * (1-t)
          if (n_vars == 2) {
            # Simple dependence measure
            log_ratios <- log(block_max_matrix[, 1] / (block_max_matrix[, 1] + block_max_matrix[, 2]))
            dependence_params[b] <- -mean(log_ratios, na.rm = TRUE)
          } else {
            # For higher dimensions, use average pairwise dependence
            pairwise_deps <- numeric(choose(n_vars, 2))
            idx <- 1
            for (i in 1:(n_vars-1)) {
              for (j in (i+1):n_vars) {
                log_ratios <- log(block_max_matrix[, i] / (block_max_matrix[, i] + block_max_matrix[, j]))
                pairwise_deps[idx] <- -mean(log_ratios, na.rm = TRUE)
                idx <- idx + 1
              }
            }
            dependence_params[b] <- mean(pairwise_deps, na.rm = TRUE)
          }
        }
      }
      
      # Take average dependence parameter
      dependence_param <- mean(dependence_params[!is.na(dependence_params)])
      
      # Ensure parameter is in valid range [0, 1]
      dependence_param <- max(0, min(1, dependence_param))
      
    }, error = function(e) {
      cat("Parameter estimation failed, using default value\n")
      dependence_param <- 0.5
    })
    
    model_results <- list(
      model_type = "logistic",
      dependence_parameter = dependence_param,
      parameters = list(alpha = dependence_param)
    )
  }
  
  # Calculate extremal coefficients
  if (n_vars == 2) {
    # For bivariate case
    extremal_coeff <- 2 - dependence_param
  } else {
    # For multivariate case, calculate pairwise extremal coefficients
    extremal_coeffs <- matrix(NA, n_vars, n_vars)
    
    for (i in 1:(n_vars-1)) {
      for (j in (i+1):n_vars) {
        # Simplified calculation
        extremal_coeffs[i, j] <- extremal_coeffs[j, i] <- 2 - dependence_param
      }
    }
    diag(extremal_coeffs) <- 1
  }
  
  # Tail dependence analysis
  # Calculate probability of joint exceedances
  threshold_quantiles <- c(0.9, 0.95, 0.99)
  joint_exceedance_probs <- numeric(length(threshold_quantiles))
  
  for (t in seq_along(threshold_quantiles)) {
    q <- threshold_quantiles[t]
    thresholds <- apply(complete_data, 2, quantile, q)
    
    # Count joint exceedances
    joint_exceedances <- apply(complete_data, 1, function(row) all(row > thresholds))
    joint_exceedance_probs[t] <- mean(joint_exceedances)
  }
  
  # Asymptotic independence test
  # Test whether variables are asymptotically independent
  # Using chi and chi-bar statistics
  if (n_vars == 2) {
    # Calculate chi statistic
    threshold_levels <- seq(0.7, 0.95, 0.05)
    chi_stats <- numeric(length(threshold_levels))
    chi_bar_stats <- numeric(length(threshold_levels))
    
    for (t in seq_along(threshold_levels)) {
      q <- threshold_levels[t]
      thresh1 <- quantile(complete_data[, 1], q)
      thresh2 <- quantile(complete_data[, 2], q)
      
      exceed1 <- complete_data[, 1] > thresh1
      exceed2 <- complete_data[, 2] > thresh2
      joint_exceed <- exceed1 & exceed2
      
      p1 <- mean(exceed1)
      p2 <- mean(exceed2)
      p12 <- mean(joint_exceed)
      
      if (p1 > 0 && p2 > 0) {
        chi_stats[t] <- (2 * log(min(p1, p2)) / log(p12)) - 1
        chi_bar_stats[t] <- (2 * log(1 - min(p1, p2)) / log(1 - p12)) - 1
      }
    }
    
    # Final chi and chi-bar values (limit as threshold increases)
    chi_final <- chi_stats[length(chi_stats)]
    chi_bar_final <- chi_bar_stats[length(chi_bar_stats)]
    
    # Interpretation
    if (chi_final > 0) {
      dependence_type <- "Asymptotic dependence"
    } else {
      dependence_type <- "Asymptotic independence"
    }
    
    asymptotic_tests <- list(
      chi = chi_final,
      chi_bar = chi_bar_final,
      dependence_type = dependence_type,
      chi_sequence = chi_stats,
      chi_bar_sequence = chi_bar_stats,
      threshold_levels = threshold_levels
    )
  } else {
    asymptotic_tests <- NULL
  }
  
  # Risk measure calculations
  # Multivariate VaR and Expected Shortfall
  confidence_levels <- c(0.95, 0.99, 0.995)
  portfolio_weights <- rep(1/n_vars, n_vars)  # Equal weights
  
  risk_measures <- list()
  
  for (alpha in confidence_levels) {
    # Simple approach: use empirical quantiles of portfolio
    portfolio_values <- as.numeric(complete_data %*% portfolio_weights)
    var_level <- quantile(portfolio_values, 1 - alpha)
    es_level <- mean(portfolio_values[portfolio_values >= var_level])
    
    risk_measures[[paste0("alpha_", alpha)]] <- list(
      var = var_level,
      expected_shortfall = es_level
    )
  }
  
  results <- list(
    model_results = model_results,
    extremal_coefficients = if (n_vars == 2) extremal_coeff else extremal_coeffs,
    joint_exceedance_probabilities = data.frame(
      threshold_quantile = threshold_quantiles,
      joint_prob = joint_exceedance_probs
    ),
    asymptotic_dependence_tests = asymptotic_tests,
    risk_measures = risk_measures,
    transformed_data = frechet_data,
    original_data = complete_data,
    marginal_method = marginal_method
  )
  
  cat("Multivariate EVT analysis completed\n")
  cat("Model type:", method, "\n")
  cat("Dependence parameter:", round(dependence_param, 4), "\n")
  if (n_vars == 2) {
    cat("Extremal coefficient:", round(extremal_coeff, 4), "\n")
    cat("Dependence type:", dependence_type, "\n")
  }
  
  return(results)
}

#' Extreme Value Backtesting
#' 
#' Validate extreme value models using backtesting procedures
#' @param data numeric vector or matrix of observations
#' @param model_type character: "GEV" or "GPD"
#' @param confidence_level numeric: confidence level for backtesting
#' @param window_size integer: rolling window size
#' @return backtesting results with validation statistics
extreme_value_backtest <- function(data, model_type = "GPD", 
                                  confidence_level = 0.01,
                                  window_size = 252) {
  
  cat("Starting extreme value model backtesting\n")
  cat("Model type:", model_type, ", Window size:", window_size, "\n")
  
  if (is.matrix(data)) {
    # For multivariate data, use first column or portfolio
    if (ncol(data) > 1) {
      portfolio_data <- rowMeans(data)  # Simple equal-weighted portfolio
      cat("Using equal-weighted portfolio for backtesting\n")
    } else {
      portfolio_data <- data[, 1]
    }
  } else {
    portfolio_data <- data
  }
  
  # Remove NA values
  clean_data <- portfolio_data[!is.na(portfolio_data)]
  n_obs <- length(clean_data)
  
  # Rolling window backtest
  n_forecasts <- n_obs - window_size
  var_forecasts <- numeric(n_forecasts)
  es_forecasts <- numeric(n_forecasts)
  backtest_dates <- (window_size + 1):n_obs
  
  cat("Running", n_forecasts, "backtesting iterations\n")
  
  for (i in seq_along(backtest_dates)) {
    if (i %% 50 == 0) cat("Processed", i, "out of", n_forecasts, "iterations\n")
    
    # Extract training window
    train_end <- backtest_dates[i] - 1
    train_start <- train_end - window_size + 1
    train_data <- clean_data[train_start:train_end]
    
    # Fit extreme value model on training data
    if (model_type == "GPD") {
      tryCatch({
        pot_model <- peaks_over_threshold(train_data, threshold = NULL)
        
        # Extract VaR forecast
        risk_measures <- pot_model$risk_measures
        var_idx <- which.min(abs(risk_measures$confidence_level - (1 - confidence_level)))
        var_forecasts[i] <- risk_measures$var[var_idx]
        es_forecasts[i] <- risk_measures$expected_shortfall[var_idx]
        
      }, error = function(e) {
        # Fallback to empirical quantile
        var_forecasts[i] <<- quantile(train_data, confidence_level)
        es_forecasts[i] <<- mean(train_data[train_data <= var_forecasts[i]])
      })
      
    } else if (model_type == "GEV") {
      tryCatch({
        gev_model <- block_maxima_analysis(train_data, block_size = 22)
        
        # Convert GEV to equivalent GPD for VaR calculation
        # This is approximate and requires care
        gev_params <- gev_model$parameters
        
        # Approximate VaR using GEV distribution
        # Transform to get VaR for the original distribution
        p <- confidence_level
        if (abs(gev_params$shape) < 1e-6) {
          # Gumbel case
          var_forecasts[i] <- gev_params$location - gev_params$scale * log(-log(1 - p))
        } else {
          # General GEV case
          var_forecasts[i] <- gev_params$location + gev_params$scale * 
            ((-log(1 - p))^(-gev_params$shape) - 1) / gev_params$shape
        }
        
        # Approximate ES
        if (gev_params$shape < 1) {
          es_forecasts[i] <- var_forecasts[i] + 
            (gev_params$scale - gev_params$shape * gev_params$location) / (1 - gev_params$shape)
        } else {
          es_forecasts[i] <- NA
        }
        
      }, error = function(e) {
        # Fallback to empirical quantile
        var_forecasts[i] <<- quantile(train_data, confidence_level)
        es_forecasts[i] <<- mean(train_data[train_data <= var_forecasts[i]])
      })
    }
  }
  
  # Calculate actual returns for backtest period
  actual_returns <- clean_data[backtest_dates]
  
  # VaR violations
  var_violations <- actual_returns < var_forecasts
  violation_rate <- mean(var_violations, na.rm = TRUE)
  expected_violation_rate <- confidence_level
  
  # Unconditional coverage test (Kupiec test)
  n_violations <- sum(var_violations, na.rm = TRUE)
  n_observations <- length(var_violations[!is.na(var_violations)])
  
  if (n_violations > 0 && n_violations < n_observations) {
    kupiec_lr <- -2 * log(
      (expected_violation_rate^n_violations * (1 - expected_violation_rate)^(n_observations - n_violations)) /
      ((n_violations/n_observations)^n_violations * 
       (1 - n_violations/n_observations)^(n_observations - n_violations))
    )
    kupiec_pvalue <- 1 - pchisq(kupiec_lr, df = 1)
  } else {
    kupiec_lr <- NA
    kupiec_pvalue <- NA
  }
  
  # Expected Shortfall backtesting
  es_violations <- actual_returns[var_violations & !is.na(es_forecasts)]
  es_forecasts_violations <- es_forecasts[var_violations & !is.na(es_forecasts)]
  
  if (length(es_violations) > 0) {
    es_relative_violations <- es_violations < es_forecasts_violations
    es_violation_rate <- mean(es_relative_violations, na.rm = TRUE)
    
    # ES test statistic (simplified)
    es_test_stat <- mean(es_violations - es_forecasts_violations, na.rm = TRUE)
  } else {
    es_violation_rate <- 0
    es_test_stat <- NA
  }
  
  # Performance metrics
  valid_idx <- !is.na(var_forecasts) & !is.na(actual_returns)
  mean_absolute_error <- mean(abs(actual_returns[valid_idx] - var_forecasts[valid_idx]))
  root_mean_squared_error <- sqrt(mean((actual_returns[valid_idx] - var_forecasts[valid_idx])^2))
  
  # Quantile loss function (asymmetric loss)
  quantile_loss <- mean(ifelse(actual_returns[valid_idx] < var_forecasts[valid_idx],
                              (confidence_level - 1) * (actual_returns[valid_idx] - var_forecasts[valid_idx]),
                              confidence_level * (actual_returns[valid_idx] - var_forecasts[valid_idx])))
  
  results <- list(
    model_type = model_type,
    confidence_level = confidence_level,
    var_forecasts = var_forecasts,
    es_forecasts = es_forecasts,
    actual_returns = actual_returns,
    var_violations = var_violations,
    violation_statistics = list(
      violation_rate = violation_rate,
      expected_violation_rate = expected_violation_rate,
      n_violations = n_violations,
      n_observations = n_observations
    ),
    statistical_tests = list(
      kupiec_lr = kupiec_lr,
      kupiec_pvalue = kupiec_pvalue
    ),
    es_validation = list(
      es_violation_rate = es_violation_rate,
      es_test_statistic = es_test_stat,
      n_es_violations = length(es_violations)
    ),
    performance_metrics = list(
      mae = mean_absolute_error,
      rmse = root_mean_squared_error,
      quantile_loss = quantile_loss
    ),
    backtest_dates = backtest_dates,
    window_size = window_size
  )
  
  cat("Backtesting completed\n")
  cat("VaR violation rate:", round(violation_rate * 100, 2), "% (expected:", 
      round(expected_violation_rate * 100, 2), "%)\n")
  if (!is.na(kupiec_pvalue)) {
    cat("Kupiec test p-value:", round(kupiec_pvalue, 4), "\n")
  }
  cat("Quantile loss:", round(quantile_loss, 6), "\n")
  
  return(results)
}

# Export main functions
cat("Extreme Value Theory Module Loaded Successfully\n")
cat("Available functions:\n")
cat("- block_maxima_analysis()\n")
cat("- peaks_over_threshold()\n")
cat("- multivariate_extremes()\n")
cat("- extreme_value_backtest()\n")