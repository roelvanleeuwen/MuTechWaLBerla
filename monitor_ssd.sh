#!/bin/bash

# Log file
LOG_FILE="monitor_output.log"

# Run in the background
while true; do
    # Get current date and time
    echo "===== $(date) =====" >> "$LOG_FILE"

    # Append the current disk usage to the log
    df -h >> "$LOG_FILE"

    # Append the iotop output to the log
    # Limit output to the first 10 lines for readability
    iotop -b -n 1 | head -n 20 >> "$LOG_FILE"

    # Add a separator line for clarity
    echo "======================" >> "$LOG_FILE"

    # Sleep for a specified interval (e.g., 5 seconds)
    sleep 1
done
