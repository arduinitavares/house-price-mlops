#!/bin/bash
set -e

# Function to run the drift detection and model retraining
run_model_check() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting drift detection and model check"
    
    # Create log directory if it doesn't exist
    mkdir -p /app/logs
    
    # Run drift detection
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Running drift detection..."
    python /app/src/monitoring/generate_drift_report.py 2>&1 | tee -a /app/logs/drift_$(date '+%Y%m%d').log
    
    # Check if retraining is needed
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking if model retraining is needed..."
    python /app/src/monitoring/auto_retrain.py 2>&1 | tee -a /app/logs/retrain_$(date '+%Y%m%d').log
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed drift detection and model check"
}

# Run in different modes based on environment variable
case "$1" in
    # Run once and exit (useful for manual or CI/CD triggered runs)
    "run-once")
        run_model_check
        exit 0
        ;;
    
    # Run once then start the cron service (default for container)
    "run-with-cron" | "")
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Initial run on container start"
        run_model_check
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting cron service for scheduled runs"
        cron -f
        ;;
        
    # Just start cron without initial run
    "cron-only")
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting cron service only"
        cron -f
        ;;
        
    # Any other command is passed to the shell
    *)
        exec "$@"
        ;;
esac