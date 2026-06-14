#!/bin/bash
# QUANTUM-FORGE Setup Script
# Automated deployment and initialization for the QUANTUM-FORGE HFT system

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="quantum-forge"
DOCKER_REGISTRY="your-registry.com"
IMAGE_TAG="latest"
NAMESPACE="quantum-forge"
ENVIRONMENT="${ENVIRONMENT:-development}"

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[QUANTUM-FORGE] ${message}${NC}"
}

print_header() {
    echo -e "${PURPLE}
                                                                                 
                             QUANTUM-FORGE SETUP                                 
                  Next-Generation HFT Intelligence System                        
                                                                                 
${NC}"
}

print_section() {
    echo -e "${CYAN}
                                                                                
 $1
                                                                                
${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_section "CHECKING PREREQUISITES"
    
    local missing_tools=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    else
        print_message $GREEN "  Docker found: $(docker --version)"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_tools+=("docker-compose")
    else
        if command -v docker-compose &> /dev/null; then
            print_message $GREEN "  Docker Compose found: $(docker-compose --version)"
        else
            print_message $GREEN "  Docker Compose found: $(docker compose version)"
        fi
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        missing_tools+=("python3")
    else
        if command -v python3 &> /dev/null; then
            print_message $GREEN "  Python found: $(python3 --version)"
        else
            print_message $GREEN "  Python found: $(python --version)"
        fi
    fi
    
    # Check R (optional)
    if command -v R &> /dev/null; then
        print_message $GREEN "  R found: $(R --version | head -n1)"
    else
        print_message $YELLOW "  R not found (optional for advanced statistics)"
    fi
    
    # Check Kubernetes tools (optional)
    if command -v kubectl &> /dev/null; then
        print_message $GREEN "  kubectl found: $(kubectl version --client --short 2>/dev/null)"
    else
        print_message $YELLOW "  kubectl not found (required for production deployment)"
    fi
    
    if command -v helm &> /dev/null; then
        print_message $GREEN "  Helm found: $(helm version --short)"
    else
        print_message $YELLOW "  Helm not found (optional for Kubernetes deployment)"
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        print_message $GREEN "  Git found: $(git --version)"
    else
        missing_tools+=("git")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_message $RED "  Missing required tools: ${missing_tools[*]}"
        print_message $RED "Please install the missing tools and run this script again."
        exit 1
    fi
    
    print_message $GREEN "All prerequisites satisfied!"
}

# Function to setup Python environment
setup_python_environment() {
    print_section "SETTING UP PYTHON ENVIRONMENT"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_message $BLUE "Creating Python virtual environment..."
        if command -v python3 &> /dev/null; then
            python3 -m venv venv
        else
            python -m venv venv
        fi
    fi
    
    # Activate virtual environment
    print_message $BLUE "Activating virtual environment..."
    source venv/bin/activate || source venv/Scripts/activate
    
    # Upgrade pip
    print_message $BLUE "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_message $BLUE "Installing Python dependencies..."
        pip install -r requirements.txt
    fi
    
    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        print_message $BLUE "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi
    
    print_message $GREEN "Python environment setup complete!"
}

# Function to setup R environment
setup_r_environment() {
    if ! command -v R &> /dev/null; then
        print_message $YELLOW "Skipping R setup (R not installed)"
        return 0
    fi
    
    print_section "SETTING UP R ENVIRONMENT"
    
    print_message $BLUE "Installing R packages..."
    
    # Create R script for package installation
    cat > install_r_packages.R << 'EOF'
# Required packages for quantitative finance
packages <- c(
    "quantmod",      # Quantitative Financial Modelling
    "PerformanceAnalytics",  # Performance and risk analytics
    "TTR",           # Technical Trading Rules
    "xts",           # eXtensible Time Series
    "zoo",           # S3 Infrastructure for Regular and Irregular Time Series
    "fGarch",        # Rmetrics - Autoregressive Conditional Heteroskedastic Modelling
    "rugarch",       # Univariate GARCH models
    "copula",        # Multivariate Dependence with Copulas
    "VineCopula",    # Statistical Inference of Vine Copulas
    "extremevalues", # Univariate Outlier Detection
    "changepoint",   # Methods for Changepoint Detection
    "bcp",           # Bayesian Analysis of Change Point Problems
    "HiddenMarkov",  # Hidden Markov Models
    "depmixS4",      # Dependent Mixture Models
    "vars",          # Vector Autoregression
    "urca",          # Unit Root and Cointegration Tests
    "forecast",      # Forecasting Functions for Time Series
    "RQuantLib",     # R Interface to the QuantLib Library
    "RcppArmadillo", # Rcpp Integration for the Armadillo Templated Linear Algebra Library
    "data.table",    # Extension of data.frame
    "dplyr",         # A Grammar of Data Manipulation
    "ggplot2",       # Create Elegant Data Visualisations Using the Grammar of Graphics
    "plotly",        # Create Interactive Web Graphics via 'plotly.js'
    "DBI",           # R Database Interface
    "RPostgreSQL",   # R Interface to the PostgreSQL Database System
    "redis",         # R driver for Redis
    "parallel",      # Support for Parallel computation in R
    "doParallel",    # Foreach Parallel Adaptor for the 'parallel' Package
    "Rcpp",          # Seamless R and C++ Integration
    "RcppEigen",     # Rcpp Integration for the Eigen Templated Linear Algebra Library
    "microbenchmark" # Accurate Timing Functions
)

# Install missing packages
new_packages <- packages[!(packages %in% rownames(installed.packages()))]
if(length(new_packages)) {
    cat("Installing R packages:", paste(new_packages, collapse=", "), "\n")
    install.packages(new_packages, repos="http://cran.r-project.org", dependencies=TRUE)
}

cat("R environment setup complete!\n")
EOF
    
    # Run R package installation
    Rscript install_r_packages.R
    rm install_r_packages.R
    
    print_message $GREEN "R environment setup complete!"
}

# Function to setup configuration
setup_configuration() {
    print_section "SETTING UP CONFIGURATION"
    
    # Create directories
    mkdir -p data/{raw,processed,features,models}
    mkdir -p logs
    mkdir -p config/firms
    mkdir -p config/strategies
    mkdir -p backtest_results
    
    # Create default configuration files
    if [ ! -f "config/system.yaml" ]; then
        print_message $BLUE "Creating default system configuration..."
        cat > config/system.yaml << EOF
# QUANTUM-FORGE System Configuration
system:
  name: "QUANTUM-FORGE"
  version: "1.0.0"
  environment: "${ENVIRONMENT}"
  
  # Database Configuration
  database:
    type: "timescaledb"
    host: "localhost"
    port: 5432
    name: "quantum_forge"
    user: "quantum_user"
    
  # Cache Configuration
  cache:
    type: "redis"
    host: "localhost"
    port: 6379
    db: 0
    
  # Logging Configuration
  logging:
    level: "INFO"
    format: "json"
    file: "logs/quantum_forge.log"
    max_size: "100MB"
    backup_count: 10
    
  # Performance Configuration
  performance:
    max_threads: 8
    batch_size: 1000
    cache_size: "1GB"
    
  # Risk Management
  risk:
    max_position_size: 1000000  # $1M
    max_drawdown: 0.05  # 5%
    var_confidence: 0.99
    
  # Data Sources
  data_sources:
    binance:
      enabled: true
      rate_limit: 1200  # requests per minute
    alpha_vantage:
      enabled: true
      rate_limit: 5  # requests per minute
EOF
    fi
    
    # Create firm configuration template
    if [ ! -f "config/firms/default.yaml" ]; then
        print_message $BLUE "Creating default firm configuration..."
        cat > config/firms/default.yaml << EOF
# Default Firm Configuration
firm:
  name: "Default Trading Firm"
  id: "default"
  
  # Capital allocation
  capital:
    total: 10000000  # $10M
    max_risk: 0.02   # 2% of capital
    
  # Trading parameters
  trading:
    enabled_strategies: ["momentum", "mean_reversion"]
    max_positions: 50
    position_sizing: "kelly"
    
  # Risk limits
  risk_limits:
    daily_var: 100000      # $100K
    max_leverage: 3.0
    concentration_limit: 0.1  # 10% per position
    
  # Execution settings
  execution:
    default_algo: "twap"
    slippage_tolerance: 0.001  # 10 bps
    latency_target: 1000       # 1ms
EOF
    fi
    
    print_message $GREEN "Configuration setup complete!"
}

# Function to build Docker images
build_docker_images() {
    print_section "BUILDING DOCKER IMAGES"
    
    # Build main application image
    print_message $BLUE "Building QUANTUM-FORGE application image..."
    docker build -t ${PROJECT_NAME}:${IMAGE_TAG} .
    
    # Tag for registry if specified
    if [ "${DOCKER_REGISTRY}" != "your-registry.com" ]; then
        docker tag ${PROJECT_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}
        print_message $GREEN "Tagged image for registry: ${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}"
    fi
    
    print_message $GREEN "Docker images built successfully!"
}

# Function to start services
start_services() {
    print_section "STARTING SERVICES"
    
    if [ "${ENVIRONMENT}" = "development" ]; then
        print_message $BLUE "Starting development environment with Docker Compose..."
        
        # Use docker-compose if available, otherwise docker compose
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d
        else
            docker compose up -d
        fi
        
        # Wait for services to be ready
        print_message $BLUE "Waiting for services to be ready..."
        sleep 30
        
        # Check service health
        print_message $BLUE "Checking service health..."
        
        # Check TimescaleDB
        until docker exec quantum-forge-timescaledb-1 pg_isready -U quantum_user -d quantum_forge 2>/dev/null; do
            print_message $YELLOW "Waiting for TimescaleDB..."
            sleep 5
        done
        print_message $GREEN "  TimescaleDB is ready"
        
        # Check Redis
        until docker exec quantum-forge-redis-1 redis-cli ping 2>/dev/null | grep -q PONG; do
            print_message $YELLOW "Waiting for Redis..."
            sleep 5
        done
        print_message $GREEN "  Redis is ready"
        
    else
        print_message $BLUE "Production deployment requires Kubernetes..."
        if command -v kubectl &> /dev/null; then
            deploy_to_kubernetes
        else
            print_message $RED "kubectl not found. Cannot deploy to production."
            exit 1
        fi
    fi
    
    print_message $GREEN "Services started successfully!"
}

# Function to deploy to Kubernetes
deploy_to_kubernetes() {
    print_section "DEPLOYING TO KUBERNETES"
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply production configuration
    if [ -f "infrastructure/deployment/production.yaml" ]; then
        print_message $BLUE "Deploying production configuration..."
        kubectl apply -f infrastructure/deployment/production.yaml
    fi
    
    # Apply monitoring configuration
    if [ -f "infrastructure/deployment/monitoring.yaml" ]; then
        print_message $BLUE "Deploying monitoring stack..."
        kubectl apply -f infrastructure/deployment/monitoring.yaml
    fi
    
    # Wait for deployment
    print_message $BLUE "Waiting for deployment to be ready..."
    kubectl rollout status deployment/quantum-forge-app -n ${NAMESPACE} --timeout=600s
    
    print_message $GREEN "Kubernetes deployment complete!"
}

# Function to initialize database
initialize_database() {
    print_section "INITIALIZING DATABASE"
    
    # Wait for database to be ready
    print_message $BLUE "Waiting for database connection..."
    
    if [ "${ENVIRONMENT}" = "development" ]; then
        # Run database initialization
        print_message $BLUE "Creating database schema..."
        docker exec quantum-forge-timescaledb-1 psql -U quantum_user -d quantum_forge -c "
            -- Create extensions
            CREATE EXTENSION IF NOT EXISTS timescaledb;
            CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
            
            -- Create tables
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                price DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                bid DECIMAL(20,8),
                ask DECIMAL(20,8),
                PRIMARY KEY (timestamp, symbol)
            );
            
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(4) NOT NULL,
                quantity DECIMAL(20,8) NOT NULL,
                price DECIMAL(20,8) NOT NULL,
                commission DECIMAL(20,8),
                strategy VARCHAR(50)
            );
            
            CREATE TABLE IF NOT EXISTS positions (
                symbol VARCHAR(20) PRIMARY KEY,
                quantity DECIMAL(20,8) NOT NULL,
                avg_price DECIMAL(20,8) NOT NULL,
                unrealized_pnl DECIMAL(20,8),
                last_updated TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Create hypertables
            SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
            SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy, timestamp DESC);
            
            COMMENT ON TABLE market_data IS 'Time-series market data storage';
            COMMENT ON TABLE trades IS 'Trade execution records';
            COMMENT ON TABLE positions IS 'Current position tracking';
        "
        
        print_message $GREEN "Database initialized successfully!"
    fi
}

# Function to run tests
run_tests() {
    print_section "RUNNING TESTS"
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    # Run Python tests
    if command -v pytest &> /dev/null; then
        print_message $BLUE "Running Python tests..."
        pytest tests/ -v --cov=. --cov-report=html --cov-report=term
    else
        print_message $YELLOW "pytest not found, skipping Python tests"
    fi
    
    # Run R tests
    if command -v R &> /dev/null && [ -d "tests/r" ]; then
        print_message $BLUE "Running R tests..."
        Rscript -e "testthat::test_dir('tests/r')"
    fi
    
    print_message $GREEN "Tests completed!"
}

# Function to display status
show_status() {
    print_section "SYSTEM STATUS"
    
    if [ "${ENVIRONMENT}" = "development" ]; then
        print_message $BLUE "Development Environment Status:"
        
        # Check Docker containers
        if command -v docker-compose &> /dev/null; then
            docker-compose ps
        else
            docker compose ps
        fi
        
        echo ""
        print_message $BLUE "Service URLs:"
        print_message $GREEN "• Application: http://localhost:8000"
        print_message $GREEN "• Streamlit Dashboard: http://localhost:8501"
        print_message $GREEN "• Grafana: http://localhost:3000 (admin/admin)"
        print_message $GREEN "• Prometheus: http://localhost:9090"
        print_message $GREEN "• Redis Insight: http://localhost:8001"
        print_message $GREEN "• PgAdmin: http://localhost:5050"
        print_message $GREEN "• Jupyter: http://localhost:8888"
        
    else
        print_message $BLUE "Production Environment Status:"
        kubectl get pods -n ${NAMESPACE}
        kubectl get services -n ${NAMESPACE}
    fi
}

# Function to cleanup
cleanup() {
    print_section "CLEANUP"
    
    if [ "${ENVIRONMENT}" = "development" ]; then
        print_message $BLUE "Stopping development services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose down -v
        else
            docker compose down -v
        fi
    else
        print_message $BLUE "Cleaning up Kubernetes resources..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    fi
    
    print_message $GREEN "Cleanup complete!"
}

# Function to show help
show_help() {
    cat << EOF
QUANTUM-FORGE Setup Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    setup       Complete system setup (default)
    start       Start services only
    stop        Stop services
    restart     Restart services
    status      Show system status
    test        Run tests
    clean       Cleanup all resources
    help        Show this help message

Options:
    --environment=ENV    Set environment (development/production)
    --registry=URL       Docker registry URL
    --tag=TAG           Docker image tag
    --skip-tests        Skip running tests
    --skip-r            Skip R setup

Examples:
    $0                                    # Full setup
    $0 start                             # Start services only
    $0 --environment=production setup    # Production setup
    $0 status                           # Check status
    $0 clean                            # Cleanup everything

EOF
}

# Parse command line arguments
COMMAND="setup"
SKIP_TESTS=false
SKIP_R=false

while [[ $# -gt 0 ]]; do
    case $1 in
        setup|start|stop|restart|status|test|clean|help)
            COMMAND="$1"
            shift
            ;;
        --environment=*)
            ENVIRONMENT="${1#*=}"
            shift
            ;;
        --registry=*)
            DOCKER_REGISTRY="${1#*=}"
            shift
            ;;
        --tag=*)
            IMAGE_TAG="${1#*=}"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-r)
            SKIP_R=true
            shift
            ;;
        *)
            print_message $RED "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header
    
    case $COMMAND in
        setup)
            check_prerequisites
            setup_python_environment
            if [ "$SKIP_R" = false ]; then
                setup_r_environment
            fi
            setup_configuration
            build_docker_images
            start_services
            initialize_database
            if [ "$SKIP_TESTS" = false ]; then
                run_tests
            fi
            show_status
            ;;
        start)
            start_services
            show_status
            ;;
        stop)
            cleanup
            ;;
        restart)
            cleanup
            start_services
            show_status
            ;;
        status)
            show_status
            ;;
        test)
            run_tests
            ;;
        clean)
            cleanup
            ;;
        help)
            show_help
            ;;
        *)
            print_message $RED "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"

print_message $GREEN "QUANTUM-FORGE setup completed successfully!"
print_message $BLUE "For documentation, visit: https://github.com/your-org/quantum-forge"
print_message $PURPLE "Happy Trading!  "