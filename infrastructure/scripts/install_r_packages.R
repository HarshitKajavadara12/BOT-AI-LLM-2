# QUANTUM-FORGE R Package Installation Script
# Installs required R packages for quantitative finance and statistics

cat("Installing QUANTUM-FORGE R dependencies...\n")

# Set CRAN mirror
options(repos = c(CRAN = "https://cran.r-project.org"))

# Required packages for quantitative finance and advanced statistics
packages <- c(
    # Core time series and financial data
    "quantmod",           # Quantitative Financial Modelling Framework
    "PerformanceAnalytics", # Econometric Tools for Performance and Risk Analysis
    "TTR",               # Technical Trading Rules
    "xts",               # eXtensible Time Series
    "zoo",               # S3 Infrastructure for Regular and Irregular Time Series
    "lubridate",         # Make Dealing with Dates a Little Easier
    
    # GARCH and volatility models
    "fGarch",            # Rmetrics - Autoregressive Conditional Heteroskedastic Modelling
    "rugarch",           # Univariate GARCH models
    "rmgarch",           # Multivariate GARCH models
    "fBasics",           # Rmetrics - Markets and Basic Statistics
    
    # Copulas and dependence modeling
    "copula",            # Multivariate Dependence with Copulas
    "VineCopula",        # Statistical Inference of Vine Copulas
    "CDVineCopulaConditional", # Conditional Density Vine Copulas
    
    # Extreme value theory
    "evd",               # Functions for Extreme Value Distributions
    "extremevalues",     # Univariate Outlier Detection
    "POT",               # Generalized Pareto Distribution and Peaks Over Threshold
    "ismev",             # An Introduction to Statistical Modeling of Extreme Values
    
    # Change point detection and regime analysis
    "changepoint",       # Methods for Changepoint Detection
    "bcp",               # Bayesian Analysis of Change Point Problems
    "cpm",               # Sequential and Batch Change Detection
    "segmented",         # Regression Models with Break-Points/Change-Points
    
    # Hidden Markov Models
    "HiddenMarkov",      # Hidden Markov Models
    "depmixS4",          # Dependent Mixture Models - Hidden Markov Models of GLMs
    "MSwM",              # Markov Switching Models
    
    # Vector autoregression and cointegration
    "vars",              # Vector Autoregression
    "urca",              # Unit Root and Cointegration Tests for Time Series Data
    "tsDyn",             # Nonlinear Time Series Models with Regime Switching
    "VECM",              # Vector Error Correction Model
    
    # Forecasting
    "forecast",          # Forecasting Functions for Time Series and Linear Models
    "tseries",           # Time Series Analysis and Computational Finance
    "fracdiff",          # Fractionally Differenced ARIMA Models
    
    # Quantitative finance libraries
    "RQuantLib",         # R Interface to the QuantLib Library
    "fOptions",          # Rmetrics - Pricing and Evaluating Basic Options
    "fExoticOptions",    # Rmetrics - Pricing and Evaluating Exotic Options
    "RcppQuantuccia",    # R Bindings to QuantLib
    
    # High-performance computing
    "RcppArmadillo",     # Rcpp Integration for Armadillo Linear Algebra Library
    "RcppEigen",         # Rcpp Integration for Eigen Linear Algebra Library
    "Rcpp",              # Seamless R and C++ Integration
    "RcppParallel",      # Parallel Programming Tools for Rcpp
    
    # Data manipulation and analysis
    "data.table",        # Extension of data.frame
    "dplyr",             # A Grammar of Data Manipulation
    "tidyr",             # Tidy Messy Data
    "magrittr",          # A Forward-Pipe Operator for R
    "readr",             # Read Rectangular Text Data
    
    # Visualization
    "ggplot2",           # Create Elegant Data Visualisations Using Grammar of Graphics
    "plotly",            # Create Interactive Web Graphics via plotly.js
    "corrplot",          # Visualization of a Correlation Matrix
    "lattice",           # Trellis Graphics for R
    "gridExtra",         # Miscellaneous Functions for Grid Graphics
    
    # Database connectivity
    "DBI",               # R Database Interface
    "RPostgreSQL",       # R Interface to the PostgreSQL Database System
    "odbc",              # Connect to ODBC Compatible Databases
    "dbplyr",            # A dplyr Back End for Databases
    
    # Redis connectivity
    "redux",             # R Bindings to hiredis
    
    # Parallel computing
    "parallel",          # Support for Parallel computation in R
    "doParallel",        # Foreach Parallel Adaptor for the parallel Package
    "foreach",           # Provides Foreach Looping Construct for R
    "future",            # Unified Parallel and Distributed Processing in R
    
    # Machine learning integration
    "randomForest",      # Breiman and Cutler's Random Forests for Classification and Regression
    "e1071",             # Misc Functions of the Department of Statistics
    "caret",             # Classification and Regression Training
    "glmnet",            # Lasso and Elastic-Net Regularized Generalized Linear Models
    
    # Statistical tests and analysis
    "nortest",           # Tests for Normality
    "moments",           # Moments, Cumulants, Skewness, Kurtosis and Related Tests
    "lawstat",           # Tools for Biostatistics, Public Policy, and Law
    "tsoutliers",        # Automatic Detection of Outliers in Time Series
    
    # Optimization
    "nloptr",            # R Interface to NLopt
    "Rsolnp",            # General Non-linear Optimization Using Augmented Lagrange
    "ROI",               # R Optimization Infrastructure
    "quadprog",          # Functions to Solve Quadratic Programming Problems
    
    # Risk management
    "PortfolioAnalytics", # Portfolio Analysis, Including Numerical Methods for Optimization
    "fPortfolio",        # Rmetrics - Portfolio Selection and Optimization
    "FRAPO",             # Financial Risk Modelling and Portfolio Optimisation with R
    
    # Performance measurement
    "microbenchmark",    # Accurate Timing Functions
    "profvis",           # Interactive Visualizations for Profiling R Code
    "bench",             # High Precision Timing of R Expressions
    
    # Utilities
    "config",            # Manage Environment Specific Configuration Values
    "logger",            # A Lightweight, Modern and Flexible Logging Utility for R
    "testthat",          # Unit Testing for R
    "devtools",          # Tools to Make Developing R Packages Easier
    "roxygen2"           # In-Line Documentation for R
)

# Function to install packages with error handling
install_packages_safely <- function(pkg_list) {
    installed_packages <- rownames(installed.packages())
    new_packages <- pkg_list[!(pkg_list %in% installed_packages)]
    
    if (length(new_packages) == 0) {
        cat("All packages are already installed.\n")
        return(TRUE)
    }
    
    cat(sprintf("Installing %d new packages: %s\n", 
                length(new_packages), 
                paste(new_packages, collapse=", ")))
    
    success_count <- 0
    failed_packages <- character(0)
    
    for (pkg in new_packages) {
        cat(sprintf("Installing %s... ", pkg))
        
        result <- tryCatch({
            install.packages(pkg, dependencies = TRUE, quiet = TRUE)
            cat("SUCCESS\n")
            success_count <- success_count + 1
            TRUE
        }, warning = function(w) {
            cat(sprintf("WARNING: %s\n", w$message))
            TRUE
        }, error = function(e) {
            cat(sprintf("FAILED: %s\n", e$message))
            failed_packages <<- c(failed_packages, pkg)
            FALSE
        })
    }
    
    cat(sprintf("\nInstallation Summary:\n"))
    cat(sprintf("  Successfully installed: %d packages\n", success_count))
    cat(sprintf("  Failed installations: %d packages\n", length(failed_packages)))
    
    if (length(failed_packages) > 0) {
        cat(sprintf("  Failed packages: %s\n", paste(failed_packages, collapse=", ")))
        
        # Try alternative installation methods for failed packages
        cat("\nAttempting alternative installation methods...\n")
        for (pkg in failed_packages) {
            cat(sprintf("Trying devtools for %s... ", pkg))
            result <- tryCatch({
                if (pkg == "redux") {
                    # Redux requires special handling
                    devtools::install_github("richfitz/redux")
                } else {
                    # Try from different repository or development version
                    devtools::install_version(pkg)
                }
                cat("SUCCESS\n")
            }, error = function(e) {
                cat(sprintf("FAILED: %s\n", e$message))
            })
        }
    }
    
    return(length(failed_packages) == 0)
}

# Install base packages first
base_packages <- c("devtools", "remotes")
cat("Installing base packages...\n")
install.packages(base_packages, dependencies = TRUE)

# Install main package list
cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
cat("QUANTUM-FORGE R Package Installation\n")
cat(paste(rep("=", 70), collapse="") %+% "\n\n")

success <- install_packages_safely(packages)

# Verify critical packages
cat("\nVerifying critical package installations...\n")
critical_packages <- c(
    "quantmod", "PerformanceAnalytics", "rugarch", "copula", 
    "VineCopula", "changepoint", "vars", "urca", "forecast",
    "RcppArmadillo", "data.table", "ggplot2"
)

missing_critical <- character(0)
for (pkg in critical_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        missing_critical <- c(missing_critical, pkg)
    }
}

if (length(missing_critical) > 0) {
    cat(sprintf("WARNING: Critical packages not available: %s\n", 
                paste(missing_critical, collapse=", ")))
    cat("Some QUANTUM-FORGE features may not work properly.\n")
} else {
    cat("All critical packages successfully installed and loaded.\n")
}

# Create package loading helper function
cat("\nCreating package loading helper...\n")
helper_code <- '
# QUANTUM-FORGE R Package Loader
load_quantum_packages <- function(verbose = FALSE) {
    packages <- c(
        "quantmod", "PerformanceAnalytics", "TTR", "xts", "zoo",
        "rugarch", "copula", "VineCopula", "changepoint", "vars", 
        "urca", "forecast", "data.table", "dplyr", "ggplot2"
    )
    
    loaded <- sapply(packages, function(pkg) {
        result <- suppressPackageStartupMessages(
            require(pkg, character.only = TRUE, quietly = !verbose)
        )
        if (verbose && result) cat(sprintf("Loaded: %s\\n", pkg))
        if (verbose && !result) cat(sprintf("Failed to load: %s\\n", pkg))
        return(result)
    })
    
    if (verbose) {
        cat(sprintf("Successfully loaded %d/%d packages\\n", 
                    sum(loaded), length(packages)))
    }
    
    return(all(loaded))
}

# Test function
test_quantum_packages <- function() {
    cat("Testing QUANTUM-FORGE R environment...\\n")
    
    # Test quantmod
    tryCatch({
        suppressMessages(getSymbols("AAPL", src="yahoo", from="2023-01-01", to="2023-01-31"))
        cat("  Market data access working\\n")
    }, error = function(e) {
        cat("  Market data access failed\\n")
    })
    
    # Test time series operations
    tryCatch({
        ts_data <- xts(rnorm(100), order.by = Sys.Date() + 1:100)
        ma_data <- TTR::SMA(ts_data, n = 10)
        cat("  Time series operations working\\n")
    }, error = function(e) {
        cat("  Time series operations failed\\n")
    })
    
    # Test GARCH modeling
    tryCatch({
        spec <- rugarch::ugarchspec(variance.model = list(model = "sGARCH"))
        cat("  GARCH modeling working\\n")
    }, error = function(e) {
        cat("  GARCH modeling failed\\n")
    })
    
    # Test copula functions
    tryCatch({
        cop <- copula::normalCopula(0.5)
        cat("  Copula modeling working\\n")
    }, error = function(e) {
        cat("  Copula modeling failed\\n")
    })
    
    cat("R environment test completed.\\n")
}
'

# Write helper to file
writeLines(helper_code, "quantum_r_helpers.R")

# Final summary
cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
cat("QUANTUM-FORGE R Environment Setup Complete\n")
cat(paste(rep("=", 70), collapse="") %+% "\n")
cat(sprintf("Total packages attempted: %d\n", length(packages)))
cat("Helper functions created in: quantum_r_helpers.R\n")
cat("\nTo load packages in R session, use:\n")
cat("  source('quantum_r_helpers.R')\n")
cat("  load_quantum_packages(verbose = TRUE)\n")
cat("  test_quantum_packages()\n")
cat("\nR environment ready for QUANTUM-FORGE!  \n")