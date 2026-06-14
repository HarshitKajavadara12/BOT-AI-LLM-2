# QUANTUM-FORGE Cointegration Analysis
# Statistical arbitrage and pairs trading research framework
# Author: QUANTUM-FORGE Research Team
# Last Updated: 2024

# Load required libraries
suppressPackageStartupMessages({
  library(urca)
  library(vars)
  library(tsDyn)
  library(egcm)
  library(VECM)
  library(MTS)
  library(forecast)
  library(tseries)
  library(adf.test)
  library(xts)
  library(zoo)
  library(quantmod)
  library(PerformanceAnalytics)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(gridExtra)
  library(corrplot)
  library(RColorBrewer)
  library(parallel)
  library(doParallel)
  library(foreach)
  library(pracma)
})

cat("QUANTUM-FORGE Cointegration Analysis Module Initialized\n")

#' Comprehensive Cointegration Analysis
#' 
#' Full cointegration analysis including tests, estimation, and diagnostics
#' @param data matrix or data.frame of time series (each column is a series)
#' @param method character: cointegration method ("engle_granger", "johansen", "phillips_ouliaris")
#' @param deterministic character: deterministic components ("none", "const", "trend", "both")
#' @param lag_selection character: method for lag selection ("AIC", "BIC", "HQ")
#' @return comprehensive cointegration analysis results
cointegration_analysis <- function(data, method = "johansen", 
                                  deterministic = "const", 
                                  lag_selection = "AIC") {
  
  cat("Performing comprehensive cointegration analysis\n")
  cat("Method:", method, ", Deterministic:", deterministic, "\n")
  
  # Ensure data is in matrix format
  if (is.data.frame(data)) {
    data_matrix <- as.matrix(data)
  } else {
    data_matrix <- data
  }
  
  # Remove rows with NA values
  complete_data <- data_matrix[complete.cases(data_matrix), ]
  n_obs <- nrow(complete_data)
  n_vars <- ncol(complete_data)
  
  if (is.null(colnames(complete_data))) {
    colnames(complete_data) <- paste0("Series", 1:n_vars)
  }
  
  series_names <- colnames(complete_data)
  cat("Analyzing", n_vars, "series with", n_obs, "observations\n")
  
  # Step 1: Unit root tests for each series
  cat("Step 1: Testing for unit roots in individual series\n")
  unit_root_results <- list()
  
  for (i in 1:n_vars) {
    series <- complete_data[, i]
    series_name <- series_names[i]
    
    cat("Testing unit root for", series_name, "\n")
    
    # ADF test
    adf_test <- tryCatch({
      ur.df(series, type = "trend", lags = 1)
    }, error = function(e) NULL)
    
    # PP test
    pp_test <- tryCatch({
      ur.pp(series, type = "Z-tau", model = "trend")
    }, error = function(e) NULL)
    
    # KPSS test
    kpss_test <- tryCatch({
      ur.kpss(series, type = "tau")
    }, error = function(e) NULL)
    
    unit_root_results[[series_name]] <- list(
      adf = adf_test,
      pp = pp_test,
      kpss = kpss_test
    )
  }
  
  # Step 2: Determine optimal lag length
  cat("Step 2: Selecting optimal lag length\n")
  
  max_lags <- min(10, floor(n_obs/10))  # Conservative maximum
  
  if (method == "johansen") {
    # Use VARselect for Johansen approach
    var_select <- tryCatch({
      VARselect(complete_data, lag.max = max_lags, type = deterministic)
    }, error = function(e) {
      cat("VARselect failed, using default lag = 2\n")
      list(selection = c(AIC = 2, HQ = 2, SC = 2, FPE = 2))
    })
    
    optimal_lag <- var_select$selection[lag_selection]
    if (is.na(optimal_lag) || optimal_lag < 1) {
      optimal_lag <- 2
    }
    
  } else {
    # For other methods, use simple approach
    aic_values <- numeric(max_lags)
    
    for (k in 1:max_lags) {
      tryCatch({
        if (n_vars == 2) {
          temp_model <- lm(complete_data[, 1] ~ complete_data[, 2] + 
                          lag(complete_data[, 1], k) + lag(complete_data[, 2], k))
          aic_values[k] <- AIC(temp_model)
        } else {
          # For multivariate case, use VAR
          temp_var <- VAR(complete_data, p = k, type = deterministic)
          aic_values[k] <- AIC(temp_var)
        }
      }, error = function(e) {
        aic_values[k] <<- Inf
      })
    }
    
    optimal_lag <- which.min(aic_values)
    if (length(optimal_lag) == 0 || optimal_lag < 1) {
      optimal_lag <- 2
    }
  }
  
  cat("Selected lag length:", optimal_lag, "\n")
  
  # Step 3: Cointegration testing
  cat("Step 3: Testing for cointegration\n")
  
  cointegration_results <- list()
  
  if (method == "engle_granger" && n_vars == 2) {
    # Engle-Granger two-step method
    cat("Performing Engle-Granger cointegration test\n")
    
    # Step 1: Estimate cointegrating regression
    y1 <- complete_data[, 1]
    y2 <- complete_data[, 2]
    
    # Long-run relationship
    coint_reg <- lm(y1 ~ y2)
    residuals <- residuals(coint_reg)
    
    # Step 2: Test residuals for unit root
    eg_test <- tryCatch({
      ur.df(residuals, type = "none", lags = optimal_lag - 1)
    }, error = function(e) NULL)
    
    # Phillips-Ouliaris test as alternative
    po_test <- tryCatch({
      po.test(complete_data[, c(1, 2)])
    }, error = function(e) NULL)
    
    cointegration_results$engle_granger <- list(
      cointegrating_regression = coint_reg,
      residuals = residuals,
      eg_test = eg_test,
      po_test = po_test,
      cointegrating_vector = c(1, -coef(coint_reg)[2])
    )
    
  } else if (method == "johansen") {
    # Johansen maximum likelihood method
    cat("Performing Johansen cointegration test\n")
    
    johansen_test <- tryCatch({
      ca.jo(complete_data, type = "trace", ecdet = deterministic, K = optimal_lag)
    }, error = function(e) {
      cat("Johansen test failed with optimal lag, trying lag = 2\n")
      tryCatch({
        ca.jo(complete_data, type = "trace", ecdet = deterministic, K = 2)
      }, error = function(e2) NULL)
    })
    
    # Eigenvalue test as well
    johansen_eigen <- tryCatch({
      ca.jo(complete_data, type = "eigen", ecdet = deterministic, K = optimal_lag)
    }, error = function(e) {
      tryCatch({
        ca.jo(complete_data, type = "eigen", ecdet = deterministic, K = 2)
      }, error = function(e2) NULL)
    })
    
    cointegration_results$johansen <- list(
      trace_test = johansen_test,
      eigenvalue_test = johansen_eigen
    )
    
  } else if (method == "phillips_ouliaris") {
    # Phillips-Ouliaris test
    cat("Performing Phillips-Ouliaris cointegration test\n")
    
    po_test <- tryCatch({
      po.test(complete_data)
    }, error = function(e) NULL)
    
    cointegration_results$phillips_ouliaris <- list(
      po_test = po_test
    )
  }
  
  # Step 4: Estimate Vector Error Correction Model (if cointegration found)
  cat("Step 4: Estimating Vector Error Correction Model\n")
  
  vecm_results <- NULL
  
  if (method == "johansen" && !is.null(cointegration_results$johansen$trace_test)) {
    tryCatch({
      # Determine number of cointegrating relationships
      johansen_result <- cointegration_results$johansen$trace_test
      
      # Simple rule: use 5% critical values
      trace_stats <- johansen_result@teststat
      critical_values <- johansen_result@cval[, 2]  # 5% level
      
      r <- sum(trace_stats > critical_values)  # Number of cointegrating relations
      cat("Number of cointegrating relationships found:", r, "\n")
      
      if (r > 0 && r < n_vars) {
        # Estimate VECM
        vecm_model <- tryCatch({
          vec2var(johansen_result, r = r)
        }, error = function(e) {
          cat("VECM estimation failed\n")
          NULL
        })
        
        if (!is.null(vecm_model)) {
          # Extract cointegrating vectors
          cointegrating_vectors <- johansen_result@V[, 1:r]
          adjustment_coefficients <- johansen_result@W[, 1:r]
          
          # Calculate error correction terms
          if (r == 1) {
            ect <- as.numeric(complete_data %*% cointegrating_vectors)
          } else {
            ect <- complete_data %*% cointegrating_vectors
          }
          
          vecm_results <- list(
            vecm_model = vecm_model,
            cointegrating_vectors = cointegrating_vectors,
            adjustment_coefficients = adjustment_coefficients,
            error_correction_terms = ect,
            n_coint_relations = r
          )
        }
      }
    }, error = function(e) {
      cat("VECM estimation failed:", e$message, "\n")
    })
  }
  
  # Step 5: Model diagnostics
  cat("Step 5: Performing model diagnostics\n")
  
  diagnostics <- list()
  
  # Residual analysis for VECM
  if (!is.null(vecm_results)) {
    vecm_residuals <- residuals(vecm_results$vecm_model)
    
    # Serial correlation test (Portmanteau test)
    serial_test <- tryCatch({
      serial.test(vecm_results$vecm_model, lags.pt = min(10, optimal_lag * 2))
    }, error = function(e) NULL)
    
    # Normality test
    normality_test <- tryCatch({
      normality.test(vecm_results$vecm_model)
    }, error = function(e) NULL)
    
    # Heteroskedasticity test
    arch_test <- tryCatch({
      arch.test(vecm_results$vecm_model, lags.multi = 5)
    }, error = function(e) NULL)
    
    diagnostics$vecm <- list(
      residuals = vecm_residuals,
      serial_correlation = serial_test,
      normality = normality_test,
      arch = arch_test
    )
  }
  
  # Step 6: Half-life of mean reversion
  cat("Step 6: Calculating half-life of mean reversion\n")
  
  half_life_results <- list()
  
  if (method == "engle_granger" && !is.null(cointegration_results$engle_granger)) {
    # For Engle-Granger, estimate AR(1) on residuals
    residuals <- cointegration_results$engle_granger$residuals
    
    # Remove first observation for AR(1)
    y_lag <- residuals[-length(residuals)]
    y_current <- residuals[-1]
    
    ar_model <- lm(y_current ~ y_lag - 1)  # No intercept for stationary series
    phi <- coef(ar_model)[1]
    
    if (phi < 1 && phi > 0) {
      half_life_eg <- -log(0.5) / log(phi)
    } else {
      half_life_eg <- Inf
    }
    
    half_life_results$engle_granger <- half_life_eg
  }
  
  if (!is.null(vecm_results)) {
    # For VECM, use adjustment coefficients
    adj_coeffs <- vecm_results$adjustment_coefficients
    
    if (is.matrix(adj_coeffs)) {
      # Multiple cointegrating relationships
      half_lives_vecm <- numeric(ncol(adj_coeffs))
      for (i in 1:ncol(adj_coeffs)) {
        alpha <- adj_coeffs[1, i]  # Speed of adjustment for first variable
        if (alpha < 0) {
          half_lives_vecm[i] <- -log(0.5) / log(1 + alpha)
        } else {
          half_lives_vecm[i] <- Inf
        }
      }
    } else {
      # Single cointegrating relationship
      alpha <- adj_coeffs[1]
      if (alpha < 0) {
        half_lives_vecm <- -log(0.5) / log(1 + alpha)
      } else {
        half_lives_vecm <- Inf
      }
    }
    
    half_life_results$vecm <- half_lives_vecm
  }
  
  # Step 7: Trading signal generation (for pairs trading)
  cat("Step 7: Generating trading signals\n")
  
  trading_signals <- NULL
  
  if (n_vars == 2 && method == "engle_granger" && 
      !is.null(cointegration_results$engle_granger)) {
    
    residuals <- cointegration_results$engle_granger$residuals
    
    # Calculate rolling statistics
    window_size <- min(60, floor(n_obs / 4))  # 60-day or 25% of data
    
    rolling_mean <- rollapply(residuals, width = window_size, FUN = mean, 
                             fill = NA, align = "right")
    rolling_sd <- rollapply(residuals, width = window_size, FUN = sd, 
                           fill = NA, align = "right")
    
    # Z-score of residuals
    z_score <- (residuals - rolling_mean) / rolling_sd
    
    # Trading signals
    # Long spread when z-score < -2, short when z-score > 2
    # Close positions when z-score crosses zero
    
    signal <- numeric(length(z_score))
    position <- 0  # Current position: 1 = long spread, -1 = short spread, 0 = no position
    
    entry_threshold <- 2.0
    exit_threshold <- 0.0
    
    for (i in 2:length(z_score)) {
      if (!is.na(z_score[i])) {
        if (position == 0) {
          # No position, look for entry signals
          if (z_score[i] > entry_threshold) {
            position <- -1  # Short spread (short series 1, long series 2)
            signal[i] <- -1
          } else if (z_score[i] < -entry_threshold) {
            position <- 1   # Long spread (long series 1, short series 2)
            signal[i] <- 1
          } else {
            signal[i] <- 0
          }
        } else {
          # Have position, look for exit signals
          if ((position == 1 && z_score[i] > exit_threshold) ||
              (position == -1 && z_score[i] < exit_threshold)) {
            position <- 0  # Close position
            signal[i] <- 0
          } else {
            signal[i] <- position  # Maintain position
          }
        }
      } else {
        signal[i] <- 0
      }
    }
    
    trading_signals <- list(
      residuals = residuals,
      z_score = z_score,
      rolling_mean = rolling_mean,
      rolling_sd = rolling_sd,
      signals = signal,
      entry_threshold = entry_threshold,
      exit_threshold = exit_threshold
    )
  }
  
  # Step 8: Performance metrics (if trading signals available)
  performance_metrics <- NULL
  
  if (!is.null(trading_signals) && n_vars == 2) {
    cat("Step 8: Calculating trading performance\n")
    
    # Calculate spread returns
    y1_returns <- diff(log(complete_data[, 1]))
    y2_returns <- diff(log(complete_data[, 2]))
    
    # Spread return based on cointegrating vector
    coint_vector <- cointegration_results$engle_granger$cointegrating_vector
    spread_returns <- coint_vector[1] * y1_returns + coint_vector[2] * y2_returns
    
    # Strategy returns based on signals
    # Use lagged signals to avoid look-ahead bias
    lagged_signals <- c(0, trading_signals$signals[-length(trading_signals$signals)])
    strategy_returns <- lagged_signals[-1] * spread_returns  # Remove first NA
    
    # Performance metrics
    total_return <- sum(strategy_returns, na.rm = TRUE)
    annualized_return <- total_return * 252 / length(strategy_returns)  # Assuming daily data
    volatility <- sd(strategy_returns, na.rm = TRUE) * sqrt(252)
    sharpe_ratio <- annualized_return / volatility
    
    # Maximum drawdown
    cumulative_returns <- cumsum(strategy_returns)
    running_max <- cummax(cumulative_returns)
    drawdown <- cumulative_returns - running_max
    max_drawdown <- min(drawdown, na.rm = TRUE)
    
    # Number of trades
    n_trades <- sum(abs(diff(lagged_signals)) > 0, na.rm = TRUE)
    
    # Win rate
    winning_trades <- sum(strategy_returns > 0, na.rm = TRUE)
    total_trades_with_returns <- sum(!is.na(strategy_returns) & strategy_returns != 0)
    win_rate <- if (total_trades_with_returns > 0) winning_trades / total_trades_with_returns else 0
    
    performance_metrics <- list(
      total_return = total_return,
      annualized_return = annualized_return,
      volatility = volatility,
      sharpe_ratio = sharpe_ratio,
      max_drawdown = max_drawdown,
      n_trades = n_trades,
      win_rate = win_rate,
      strategy_returns = strategy_returns,
      cumulative_returns = cumulative_returns
    )
  }
  
  # Compile final results
  results <- list(
    method = method,
    data_info = list(
      n_obs = n_obs,
      n_vars = n_vars,
      series_names = series_names
    ),
    unit_root_tests = unit_root_results,
    optimal_lag = optimal_lag,
    cointegration_results = cointegration_results,
    vecm_results = vecm_results,
    diagnostics = diagnostics,
    half_life = half_life_results,
    trading_signals = trading_signals,
    performance_metrics = performance_metrics,
    original_data = complete_data
  )
  
  cat("Cointegration analysis completed\n")
  
  # Summary output
  if (method == "johansen" && !is.null(vecm_results)) {
    cat("Number of cointegrating relationships:", vecm_results$n_coint_relations, "\n")
    if (!is.null(half_life_results$vecm)) {
      cat("Half-life of mean reversion:", round(mean(half_life_results$vecm[is.finite(half_life_results$vecm)]), 2), "periods\n")
    }
  } else if (method == "engle_granger" && !is.null(half_life_results$engle_granger)) {
    cat("Half-life of mean reversion:", round(half_life_results$engle_granger, 2), "periods\n")
  }
  
  if (!is.null(performance_metrics)) {
    cat("Trading performance - Sharpe ratio:", round(performance_metrics$sharpe_ratio, 3), "\n")
    cat("Win rate:", round(performance_metrics$win_rate * 100, 1), "%\n")
  }
  
  return(results)
}

#' Pairs Trading Strategy
#' 
#' Automated pairs trading strategy based on cointegration
#' @param data matrix of price series (2 columns for pair)
#' @param formation_period integer: number of periods for cointegration estimation
#' @param trading_period integer: number of periods for out-of-sample trading
#' @param entry_threshold numeric: z-score threshold for position entry
#' @param exit_threshold numeric: z-score threshold for position exit
#' @return pairs trading strategy results
pairs_trading_strategy <- function(data, formation_period = 252, trading_period = 63,
                                  entry_threshold = 2.0, exit_threshold = 0.5) {
  
  cat("Implementing pairs trading strategy\n")
  cat("Formation period:", formation_period, ", Trading period:", trading_period, "\n")
  
  if (ncol(data) != 2) {
    stop("Pairs trading requires exactly 2 time series")
  }
  
  # Ensure no NA values
  complete_data <- data[complete.cases(data), ]
  n_obs <- nrow(complete_data)
  
  if (n_obs < formation_period + trading_period) {
    stop("Insufficient data for formation and trading periods")
  }
  
  # Split data into formation and trading periods
  formation_data <- complete_data[1:formation_period, ]
  trading_data <- complete_data[(formation_period + 1):(formation_period + trading_period), ]
  
  # Step 1: Formation period - estimate cointegration relationship
  cat("Step 1: Estimating cointegration in formation period\n")
  
  formation_analysis <- cointegration_analysis(formation_data, method = "engle_granger")
  
  # Check if cointegration exists
  if (is.null(formation_analysis$cointegration_results$engle_granger)) {
    cat("No cointegration found in formation period\n")
    return(NULL)
  }
  
  # Extract cointegrating relationship
  coint_reg <- formation_analysis$cointegration_results$engle_granger$cointegrating_regression
  beta <- coef(coint_reg)[2]  # Hedge ratio
  
  cat("Estimated hedge ratio:", round(beta, 4), "\n")
  
  # Step 2: Calculate spread in formation period for parameter estimation
  y1_formation <- formation_data[, 1]
  y2_formation <- formation_data[, 2]
  spread_formation <- y1_formation - beta * y2_formation
  
  # Estimate spread parameters (mean and standard deviation)
  spread_mean <- mean(spread_formation)
  spread_sd <- sd(spread_formation)
  
  # Step 3: Generate trading signals in trading period
  cat("Step 2: Generating trading signals in trading period\n")
  
  y1_trading <- trading_data[, 1]
  y2_trading <- trading_data[, 2]
  spread_trading <- y1_trading - beta * y2_trading
  
  # Standardized spread (z-score)
  z_score <- (spread_trading - spread_mean) / spread_sd
  
  # Generate trading signals
  signals <- numeric(length(z_score))
  positions <- numeric(length(z_score))
  current_position <- 0
  
  for (i in 1:length(z_score)) {
    if (current_position == 0) {
      # No position - look for entry signals
      if (z_score[i] > entry_threshold) {
        current_position <- -1  # Short spread (short stock 1, long stock 2)
        signals[i] <- -1
      } else if (z_score[i] < -entry_threshold) {
        current_position <- 1   # Long spread (long stock 1, short stock 2)
        signals[i] <- 1
      }
    } else {
      # Have position - look for exit signals
      if ((current_position == 1 && z_score[i] > -exit_threshold) ||
          (current_position == -1 && z_score[i] < exit_threshold)) {
        current_position <- 0
        signals[i] <- 0
      } else {
        signals[i] <- current_position
      }
    }
    positions[i] <- current_position
  }
  
  # Step 4: Calculate returns
  cat("Step 3: Calculating strategy returns\n")
  
  # Individual stock returns
  ret1 <- diff(log(y1_trading))
  ret2 <- diff(log(y2_trading))
  
  # Spread returns
  spread_returns <- ret1 - beta * ret2
  
  # Strategy returns (use lagged positions to avoid look-ahead bias)
  lagged_positions <- c(0, positions[-length(positions)])
  strategy_returns <- lagged_positions[-1] * spread_returns
  
  # Individual leg returns for position tracking
  leg1_returns <- lagged_positions[-1] * ret1  # Long/short stock 1
  leg2_returns <- -lagged_positions[-1] * beta * ret2  # Short/long stock 2 * hedge ratio
  
  # Step 5: Performance analysis
  cat("Step 4: Analyzing performance\n")
  
  # Cumulative returns
  cumulative_strategy <- cumsum(strategy_returns)
  cumulative_leg1 <- cumsum(leg1_returns)
  cumulative_leg2 <- cumsum(leg2_returns)
  
  # Performance metrics
  total_return <- sum(strategy_returns, na.rm = TRUE)
  n_trading_days <- length(strategy_returns)
  annualized_return <- total_return * 252 / n_trading_days
  
  volatility <- sd(strategy_returns, na.rm = TRUE) * sqrt(252)
  sharpe_ratio <- if (volatility > 0) annualized_return / volatility else 0
  
  # Maximum drawdown
  running_max <- cummax(cumulative_strategy)
  drawdown <- cumulative_strategy - running_max
  max_drawdown <- min(drawdown, na.rm = TRUE)
  
  # Trade analysis
  position_changes <- diff(c(0, positions))
  entry_points <- which(position_changes != 0)
  exit_points <- which(positions[-length(positions)] != 0 & positions[-1] == 0) + 1
  
  # Match entries and exits
  if (length(entry_points) > 0 && length(exit_points) > 0) {
    n_trades <- min(length(entry_points), length(exit_points))
    
    trade_returns <- numeric(n_trades)
    trade_durations <- numeric(n_trades)
    
    for (i in 1:n_trades) {
      if (i <= length(exit_points)) {
        entry_idx <- entry_points[i]
        exit_idx <- exit_points[i]
        
        trade_returns[i] <- sum(strategy_returns[entry_idx:(exit_idx-1)], na.rm = TRUE)
        trade_durations[i] <- exit_idx - entry_idx
      }
    }
    
    winning_trades <- sum(trade_returns > 0)
    win_rate <- winning_trades / n_trades
    avg_win <- if (winning_trades > 0) mean(trade_returns[trade_returns > 0]) else 0
    avg_loss <- if (n_trades - winning_trades > 0) mean(trade_returns[trade_returns <= 0]) else 0
    profit_factor <- if (abs(avg_loss) > 0) (winning_trades * avg_win) / ((n_trades - winning_trades) * abs(avg_loss)) else Inf
    
  } else {
    n_trades <- 0
    win_rate <- 0
    avg_win <- 0
    avg_loss <- 0
    profit_factor <- 0
    trade_returns <- numeric(0)
    trade_durations <- numeric(0)
  }
  
  # Risk metrics
  var_95 <- quantile(strategy_returns, 0.05, na.rm = TRUE)
  var_99 <- quantile(strategy_returns, 0.01, na.rm = TRUE)
  
  # Calmar ratio (annualized return / max drawdown)
  calmar_ratio <- if (abs(max_drawdown) > 0) abs(annualized_return / max_drawdown) else 0
  
  # Step 6: Statistical tests on strategy returns
  cat("Step 5: Performing statistical tests\n")
  
  # Jarque-Bera test for normality
  jb_test <- tryCatch({
    jarque.bera.test(strategy_returns)
  }, error = function(e) NULL)
  
  # Ljung-Box test for serial correlation
  lb_test <- tryCatch({
    Box.test(strategy_returns, lag = 10, type = "Ljung-Box")
  }, error = function(e) NULL)
  
  # ARCH test for heteroskedasticity
  arch_test <- tryCatch({
    archTest(strategy_returns, lags = 5)
  }, error = function(e) NULL)
  
  results <- list(
    formation_analysis = formation_analysis,
    strategy_parameters = list(
      formation_period = formation_period,
      trading_period = trading_period,
      hedge_ratio = beta,
      spread_mean = spread_mean,
      spread_sd = spread_sd,
      entry_threshold = entry_threshold,
      exit_threshold = exit_threshold
    ),
    trading_data = list(
      spread = spread_trading,
      z_score = z_score,
      signals = signals,
      positions = positions
    ),
    returns = list(
      strategy_returns = strategy_returns,
      leg1_returns = leg1_returns,
      leg2_returns = leg2_returns,
      cumulative_strategy = cumulative_strategy,
      cumulative_leg1 = cumulative_leg1,
      cumulative_leg2 = cumulative_leg2
    ),
    performance_metrics = list(
      total_return = total_return,
      annualized_return = annualized_return,
      volatility = volatility,
      sharpe_ratio = sharpe_ratio,
      max_drawdown = max_drawdown,
      calmar_ratio = calmar_ratio,
      var_95 = var_95,
      var_99 = var_99
    ),
    trade_analysis = list(
      n_trades = n_trades,
      win_rate = win_rate,
      avg_win = avg_win,
      avg_loss = avg_loss,
      profit_factor = profit_factor,
      trade_returns = trade_returns,
      trade_durations = trade_durations
    ),
    statistical_tests = list(
      jarque_bera = jb_test,
      ljung_box = lb_test,
      arch = arch_test
    ),
    original_data = complete_data
  )
  
  # Summary output
  cat("Pairs trading strategy completed\n")
  cat("Number of trades:", n_trades, "\n")
  cat("Win rate:", round(win_rate * 100, 1), "%\n")
  cat("Sharpe ratio:", round(sharpe_ratio, 3), "\n")
  cat("Max drawdown:", round(max_drawdown * 100, 2), "%\n")
  cat("Calmar ratio:", round(calmar_ratio, 3), "\n")
  
  return(results)
}

#' Multi-pair Portfolio Strategy
#' 
#' Portfolio of multiple pairs trading strategies
#' @param data_list list of matrices, each containing a pair of price series
#' @param formation_period integer: formation period for each pair
#' @param trading_period integer: trading period
#' @param correlation_threshold numeric: maximum correlation between pairs
#' @return multi-pair portfolio results
multi_pair_portfolio <- function(data_list, formation_period = 252, 
                                trading_period = 63, correlation_threshold = 0.3) {
  
  cat("Implementing multi-pair portfolio strategy\n")
  cat("Number of pairs:", length(data_list), "\n")
  
  # Step 1: Run pairs trading for each pair
  pair_results <- list()
  pair_returns <- list()
  
  for (i in 1:length(data_list)) {
    cat("Processing pair", i, "\n")
    
    pair_data <- data_list[[i]]
    if (ncol(pair_data) != 2) {
      cat("Warning: Pair", i, "does not have exactly 2 series, skipping\n")
      next
    }
    
    # Run pairs trading strategy
    pair_result <- tryCatch({
      pairs_trading_strategy(pair_data, formation_period, trading_period)
    }, error = function(e) {
      cat("Error processing pair", i, ":", e$message, "\n")
      NULL
    })
    
    if (!is.null(pair_result)) {
      pair_results[[i]] <- pair_result
      pair_returns[[i]] <- pair_result$returns$strategy_returns
    }
  }
  
  # Remove NULL results
  valid_pairs <- !sapply(pair_results, is.null)
  pair_results <- pair_results[valid_pairs]
  pair_returns <- pair_returns[valid_pairs]
  
  if (length(pair_results) == 0) {
    cat("No valid pairs found\n")
    return(NULL)
  }
  
  cat("Successfully processed", length(pair_results), "pairs\n")
  
  # Step 2: Filter pairs based on correlation
  if (length(pair_returns) > 1) {
    # Calculate correlation matrix of pair returns
    min_length <- min(sapply(pair_returns, length))
    returns_matrix <- matrix(NA, nrow = min_length, ncol = length(pair_returns))
    
    for (i in 1:length(pair_returns)) {
      returns_matrix[, i] <- pair_returns[[i]][1:min_length]
    }
    
    correlation_matrix <- cor(returns_matrix, use = "complete.obs")
    
    # Select pairs with correlation below threshold
    selected_pairs <- c(1)  # Always include first pair
    
    for (i in 2:length(pair_returns)) {
      max_corr_with_selected <- max(abs(correlation_matrix[i, selected_pairs]))
      if (max_corr_with_selected < correlation_threshold) {
        selected_pairs <- c(selected_pairs, i)
      }
    }
    
    cat("Selected", length(selected_pairs), "pairs after correlation filtering\n")
    
    # Filter results
    pair_results <- pair_results[selected_pairs]
    pair_returns <- pair_returns[selected_pairs]
  }
  
  # Step 3: Create equal-weighted portfolio
  n_pairs <- length(pair_returns)
  weights <- rep(1/n_pairs, n_pairs)
  
  # Align returns (use shortest series)
  min_length <- min(sapply(pair_returns, length))
  portfolio_returns <- numeric(min_length)
  
  for (i in 1:min_length) {
    day_returns <- sapply(1:n_pairs, function(j) {
      if (i <= length(pair_returns[[j]])) {
        pair_returns[[j]][i]
      } else {
        0
      }
    })
    portfolio_returns[i] <- sum(weights * day_returns, na.rm = TRUE)
  }
  
  # Step 4: Portfolio performance analysis
  cat("Analyzing portfolio performance\n")
  
  # Performance metrics
  total_return <- sum(portfolio_returns, na.rm = TRUE)
  annualized_return <- total_return * 252 / length(portfolio_returns)
  volatility <- sd(portfolio_returns, na.rm = TRUE) * sqrt(252)
  sharpe_ratio <- if (volatility > 0) annualized_return / volatility else 0
  
  # Maximum drawdown
  cumulative_returns <- cumsum(portfolio_returns)
  running_max <- cummax(cumulative_returns)
  drawdown <- cumulative_returns - running_max
  max_drawdown <- min(drawdown, na.rm = TRUE)
  
  # Calmar ratio
  calmar_ratio <- if (abs(max_drawdown) > 0) abs(annualized_return / max_drawdown) else 0
  
  # Risk metrics
  var_95 <- quantile(portfolio_returns, 0.05, na.rm = TRUE)
  var_99 <- quantile(portfolio_returns, 0.01, na.rm = TRUE)
  
  # Skewness and kurtosis
  skewness <- moments::skewness(portfolio_returns, na.rm = TRUE)
  kurtosis <- moments::kurtosis(portfolio_returns, na.rm = TRUE)
  
  # Individual pair performance summary
  individual_performance <- data.frame(
    pair = 1:length(pair_results),
    sharpe_ratio = sapply(pair_results, function(x) x$performance_metrics$sharpe_ratio),
    max_drawdown = sapply(pair_results, function(x) x$performance_metrics$max_drawdown),
    n_trades = sapply(pair_results, function(x) x$trade_analysis$n_trades),
    win_rate = sapply(pair_results, function(x) x$trade_analysis$win_rate)
  )
  
  # Portfolio diversification benefit
  # Compare portfolio volatility to average individual volatility
  individual_vols <- sapply(pair_results, function(x) x$performance_metrics$volatility)
  avg_individual_vol <- mean(individual_vols, na.rm = TRUE)
  diversification_ratio <- avg_individual_vol / volatility
  
  results <- list(
    individual_results = pair_results,
    portfolio_weights = weights,
    portfolio_returns = portfolio_returns,
    cumulative_returns = cumulative_returns,
    performance_metrics = list(
      total_return = total_return,
      annualized_return = annualized_return,
      volatility = volatility,
      sharpe_ratio = sharpe_ratio,
      max_drawdown = max_drawdown,
      calmar_ratio = calmar_ratio,
      var_95 = var_95,
      var_99 = var_99,
      skewness = skewness,
      kurtosis = kurtosis
    ),
    individual_performance = individual_performance,
    diversification_metrics = list(
      correlation_threshold = correlation_threshold,
      diversification_ratio = diversification_ratio,
      n_pairs_selected = n_pairs,
      avg_individual_volatility = avg_individual_vol
    ),
    formation_period = formation_period,
    trading_period = trading_period
  )
  
  cat("Multi-pair portfolio completed\n")
  cat("Portfolio Sharpe ratio:", round(sharpe_ratio, 3), "\n")
  cat("Diversification ratio:", round(diversification_ratio, 2), "\n")
  cat("Average individual Sharpe:", round(mean(individual_performance$sharpe_ratio, na.rm = TRUE), 3), "\n")
  
  return(results)
}

# Export main functions
cat("Cointegration Analysis Module Loaded Successfully\n")
cat("Available functions:\n")
cat("- cointegration_analysis()\n")
cat("- pairs_trading_strategy()\n")
cat("- multi_pair_portfolio()\n")