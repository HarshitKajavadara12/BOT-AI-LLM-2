# QUANTUM-FORGE Setup Script (PowerShell)
# Automated deployment and initialization for Windows systems

param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "start", "stop", "restart", "status", "test", "clean", "help")]
    [string]$Command = "setup",
    
    [Parameter()]
    [ValidateSet("development", "production")]
    [string]$Environment = "development",
    
    [Parameter()]
    [string]$Registry = "your-registry.com",
    
    [Parameter()]
    [string]$Tag = "latest",
    
    [Parameter()]
    [switch]$SkipTests,
    
    [Parameter()]
    [switch]$SkipR
)

# Configuration
$ProjectName = "quantum-forge"
$Namespace = "quantum-forge"

# Color functions for output
function Write-ColorOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        
        [Parameter()]
        [ConsoleColor]$ForegroundColor = [ConsoleColor]::White
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [QUANTUM-FORGE] $Message" -ForegroundColor $ForegroundColor
}

function Write-Success { 
    param([string]$Message)
    Write-ColorOutput -Message $Message -ForegroundColor Green 
}

function Write-Warning { 
    param([string]$Message)
    Write-ColorOutput -Message $Message -ForegroundColor Yellow 
}

function Write-Error { 
    param([string]$Message)
    Write-ColorOutput -Message $Message -ForegroundColor Red 
}

function Write-Info { 
    param([string]$Message)
    Write-ColorOutput -Message $Message -ForegroundColor Cyan 
}

function Write-Header {
    Write-Host @"

                                                                                 
                             QUANTUM-FORGE SETUP                                 
                  Next-Generation HFT Intelligence System                        
                                                                                 

"@ -ForegroundColor Magenta
}

function Write-Section {
    param([string]$Title)
    Write-Host @"

                                                                                
 $Title
                                                                                

"@ -ForegroundColor Cyan
}

function Test-CommandExists {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Test-Prerequisites {
    Write-Section "CHECKING PREREQUISITES"
    
    $missingTools = @()
    
    # Check Docker
    if (Test-CommandExists "docker") {
        $dockerVersion = docker --version
        Write-Success "  Docker found: $dockerVersion"
    } else {
        $missingTools += "docker"
    }
    
    # Check Docker Compose
    if ((Test-CommandExists "docker-compose") -or (docker compose version 2>$null)) {
        if (Test-CommandExists "docker-compose") {
            $composeVersion = docker-compose --version
            Write-Success "  Docker Compose found: $composeVersion"
        } else {
            $composeVersion = docker compose version
            Write-Success "  Docker Compose found: $composeVersion"
        }
    } else {
        $missingTools += "docker-compose"
    }
    
    # Check Python
    if (Test-CommandExists "python") {
        $pythonVersion = python --version
        Write-Success "  Python found: $pythonVersion"
    } elseif (Test-CommandExists "python3") {
        $pythonVersion = python3 --version
        Write-Success "  Python found: $pythonVersion"
    } else {
        $missingTools += "python"
    }
    
    # Check R (optional)
    if (Test-CommandExists "R") {
        try {
            $rVersion = & R --version | Select-Object -First 1
            Write-Success "  R found: $rVersion"
        } catch {
            Write-Warning "  R installation detected but version check failed"
        }
    } else {
        Write-Warning "  R not found (optional for advanced statistics)"
    }
    
    # Check Git
    if (Test-CommandExists "git") {
        $gitVersion = git --version
        Write-Success "  Git found: $gitVersion"
    } else {
        $missingTools += "git"
    }
    
    # Check PowerShell version
    $psVersion = $PSVersionTable.PSVersion
    if ($psVersion.Major -ge 5) {
        Write-Success "  PowerShell $($psVersion.Major).$($psVersion.Minor) found"
    } else {
        Write-Warning "  PowerShell version $($psVersion.Major).$($psVersion.Minor) may have compatibility issues"
    }
    
    if ($missingTools.Count -gt 0) {
        Write-Error "  Missing required tools: $($missingTools -join ', ')"
        Write-Error "Please install the missing tools and run this script again."
        exit 1
    }
    
    Write-Success "All prerequisites satisfied!"
}

function Set-PythonEnvironment {
    Write-Section "SETTING UP PYTHON ENVIRONMENT"
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Write-Info "Creating Python virtual environment..."
        if (Test-CommandExists "python") {
            python -m venv venv
        } else {
            python3 -m venv venv
        }
    }
    
    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    } elseif (Test-Path "venv\Scripts\activate.bat") {
        & "venv\Scripts\activate.bat"
    } else {
        Write-Error "Could not find virtual environment activation script"
        return $false
    }
    
    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install requirements
    if (Test-Path "requirements.txt") {
        Write-Info "Installing Python dependencies..."
        pip install -r requirements.txt
    }
    
    # Install development dependencies
    if (Test-Path "requirements-dev.txt") {
        Write-Info "Installing development dependencies..."
        pip install -r requirements-dev.txt
    }
    
    Write-Success "Python environment setup complete!"
    return $true
}

function Set-REnvironment {
    if (-not (Test-CommandExists "R") -or $SkipR) {
        if ($SkipR) {
            Write-Warning "Skipping R setup (--SkipR specified)"
        } else {
            Write-Warning "Skipping R setup (R not installed)"
        }
        return $true
    }
    
    Write-Section "SETTING UP R ENVIRONMENT"
    
    Write-Info "Installing R packages..."
    
    if (Test-Path "infrastructure\scripts\install_r_packages.R") {
        try {
            & R --vanilla -f "infrastructure\scripts\install_r_packages.R"
            Write-Success "R packages installed successfully!"
        } catch {
            Write-Warning "R package installation encountered issues: $($_.Exception.Message)"
        }
    } else {
        Write-Warning "R package installation script not found"
    }
    
    Write-Success "R environment setup complete!"
    return $true
}

function Set-Configuration {
    Write-Section "SETTING UP CONFIGURATION"
    
    # Create directories
    $directories = @(
        "data\raw",
        "data\processed", 
        "data\features",
        "data\models",
        "logs",
        "config\firms",
        "config\strategies",
        "backtest_results"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    # Create default system configuration
    if (-not (Test-Path "config\system.yaml")) {
        Write-Info "Creating default system configuration..."
        
        $configContent = @"
# QUANTUM-FORGE System Configuration
system:
  name: "QUANTUM-FORGE"
  version: "1.0.0"
  environment: "$Environment"
  
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
    max_position_size: 1000000  # `$1M
    max_drawdown: 0.05  # 5%
    var_confidence: 0.99
"@
        
        Set-Content -Path "config\system.yaml" -Value $configContent -Encoding UTF8
    }
    
    # Create default firm configuration
    if (-not (Test-Path "config\firms\default.yaml")) {
        Write-Info "Creating default firm configuration..."
        
        $firmContent = @"
# Default Firm Configuration
firm:
  name: "Default Trading Firm"
  id: "default"
  
  # Capital allocation
  capital:
    total: 10000000  # `$10M
    max_risk: 0.02   # 2% of capital
    
  # Trading parameters
  trading:
    enabled_strategies: ["momentum", "mean_reversion"]
    max_positions: 50
    position_sizing: "kelly"
    
  # Risk limits
  risk_limits:
    daily_var: 100000      # `$100K
    max_leverage: 3.0
    concentration_limit: 0.1  # 10% per position
"@
        
        Set-Content -Path "config\firms\default.yaml" -Value $firmContent -Encoding UTF8
    }
    
    Write-Success "Configuration setup complete!"
}

function Build-DockerImages {
    Write-Section "BUILDING DOCKER IMAGES"
    
    Write-Info "Building QUANTUM-FORGE application image..."
    docker build -t "${ProjectName}:${Tag}" .
    
    if ($Registry -ne "your-registry.com") {
        docker tag "${ProjectName}:${Tag}" "${Registry}/${ProjectName}:${Tag}"
        Write-Success "Tagged image for registry: ${Registry}/${ProjectName}:${Tag}"
    }
    
    Write-Success "Docker images built successfully!"
}

function Start-Services {
    Write-Section "STARTING SERVICES"
    
    if ($Environment -eq "development") {
        Write-Info "Starting development environment with Docker Compose..."
        
        if (Test-CommandExists "docker-compose") {
            docker-compose up -d
        } else {
            docker compose up -d
        }
        
        Write-Info "Waiting for services to be ready..."
        Start-Sleep -Seconds 30
        
        Write-Info "Checking service health..."
        
        # Check TimescaleDB
        $maxRetries = 12
        $retryCount = 0
        do {
            $retryCount++
            Write-Info "Checking TimescaleDB (attempt $retryCount)..."
            $dbReady = docker exec quantum-forge-timescaledb-1 pg_isready -U quantum_user -d quantum_forge 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "  TimescaleDB is ready"
                break
            }
            Start-Sleep -Seconds 5
        } while ($retryCount -lt $maxRetries)
        
        # Check Redis
        $retryCount = 0
        do {
            $retryCount++
            Write-Info "Checking Redis (attempt $retryCount)..."
            $redisReady = docker exec quantum-forge-redis-1 redis-cli ping 2>$null
            if ($redisReady -eq "PONG") {
                Write-Success "  Redis is ready"
                break
            }
            Start-Sleep -Seconds 5
        } while ($retryCount -lt $maxRetries)
        
        Write-Success "Services started successfully!"
    } else {
        Write-Info "Production deployment requires Kubernetes..."
        Write-Warning "Kubernetes deployment not implemented in PowerShell version"
        Write-Warning "Use the bash version (setup.sh) for production deployment"
    }
}

function Initialize-Database {
    Write-Section "INITIALIZING DATABASE"
    
    if ($Environment -eq "development") {
        Write-Info "Creating database schema..."
        
        $sqlScript = @"
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
"@
        
        try {
            $sqlScript | docker exec -i quantum-forge-timescaledb-1 psql -U quantum_user -d quantum_forge
            Write-Success "Database initialized successfully!"
        } catch {
            Write-Error "Database initialization failed: $($_.Exception.Message)"
        }
    }
}

function Invoke-Tests {
    if ($SkipTests) {
        Write-Warning "Skipping tests (--SkipTests specified)"
        return
    }
    
    Write-Section "RUNNING TESTS"
    
    # Activate virtual environment for tests
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    }
    
    # Run Python tests
    if (Test-CommandExists "pytest") {
        Write-Info "Running Python tests..."
        try {
            pytest tests\ -v --cov=. --cov-report=html --cov-report=term
            Write-Success "Python tests completed!"
        } catch {
            Write-Warning "Python tests encountered issues: $($_.Exception.Message)"
        }
    } else {
        Write-Warning "pytest not found, skipping Python tests"
    }
    
    # Run R tests
    if ((Test-CommandExists "R") -and (Test-Path "tests\r")) {
        Write-Info "Running R tests..."
        try {
            & R -e "testthat::test_dir('tests/r')"
            Write-Success "R tests completed!"
        } catch {
            Write-Warning "R tests encountered issues: $($_.Exception.Message)"
        }
    }
}

function Show-Status {
    Write-Section "SYSTEM STATUS"
    
    if ($Environment -eq "development") {
        Write-Info "Development Environment Status:"
        
        if (Test-CommandExists "docker-compose") {
            docker-compose ps
        } else {
            docker compose ps
        }
        
        Write-Host ""
        Write-Info "Service URLs:"
        Write-Success "• Application: http://localhost:8000"
        Write-Success "• Streamlit Dashboard: http://localhost:8501"
        Write-Success "• Grafana: http://localhost:3000 (admin/admin)"
        Write-Success "• Prometheus: http://localhost:9090"
        Write-Success "• Redis Insight: http://localhost:8001"
        Write-Success "• PgAdmin: http://localhost:5050"
        Write-Success "• Jupyter: http://localhost:8888"
    } else {
        Write-Info "Production Environment Status:"
        Write-Warning "Use kubectl commands to check production status"
    }
}

function Stop-Services {
    Write-Section "CLEANUP"
    
    if ($Environment -eq "development") {
        Write-Info "Stopping development services..."
        if (Test-CommandExists "docker-compose") {
            docker-compose down -v
        } else {
            docker compose down -v
        }
    } else {
        Write-Warning "Production cleanup requires kubectl commands"
    }
    
    Write-Success "Cleanup complete!"
}

function Show-Help {
    Write-Host @"
QUANTUM-FORGE Setup Script (PowerShell)

Usage: .\setup.ps1 [COMMAND] [OPTIONS]

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
    -Environment    Set environment (development/production)
    -Registry       Docker registry URL
    -Tag           Docker image tag
    -SkipTests     Skip running tests
    -SkipR         Skip R setup

Examples:
    .\setup.ps1                                    # Full setup
    .\setup.ps1 start                             # Start services only
    .\setup.ps1 -Environment production setup     # Production setup
    .\setup.ps1 status                           # Check status
    .\setup.ps1 clean                            # Cleanup everything

"@
}

# Main execution logic
function Invoke-Main {
    Write-Header
    
    switch ($Command) {
        "setup" {
            Test-Prerequisites
            Set-PythonEnvironment
            Set-REnvironment
            Set-Configuration
            Build-DockerImages
            Start-Services
            Initialize-Database
            Invoke-Tests
            Show-Status
        }
        "start" {
            Start-Services
            Show-Status
        }
        "stop" {
            Stop-Services
        }
        "restart" {
            Stop-Services
            Start-Services
            Show-Status
        }
        "status" {
            Show-Status
        }
        "test" {
            Invoke-Tests
        }
        "clean" {
            Stop-Services
        }
        "help" {
            Show-Help
        }
        default {
            Write-Error "Unknown command: $Command"
            Show-Help
            exit 1
        }
    }
}

# Error handling
try {
    Invoke-Main
    Write-Success "QUANTUM-FORGE setup completed successfully!"
    Write-Info "For documentation, visit: https://github.com/your-org/quantum-forge"
    Write-Host "Happy Trading!  " -ForegroundColor Magenta
} catch {
    Write-Error "Setup failed: $($_.Exception.Message)"
    Write-Error "Stack trace: $($_.ScriptStackTrace)"
    exit 1
} finally {
    # Cleanup logic if needed
}