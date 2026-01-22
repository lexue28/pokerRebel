#!/bin/bash
# Helper script to find Slurm logs for a ReBeL experiment

# Get current directory
CURRENT_DIR=$(pwd)

# Check if we're in an experiment directory
if [ -f "heyhi.jobid" ]; then
    JOB_ID=$(cat heyhi.jobid)
    echo "Found job ID: $JOB_ID"
    echo ""
    
    # Check if slurm directory exists
    if [ -d "slurm" ]; then
        echo "Slurm directory exists! Logs are in:"
        echo "  $(pwd)/slurm/"
        echo ""
        echo "Log files:"
        ls -lh slurm/ 2>/dev/null || echo "  (no log files yet)"
        echo ""
        echo "To view logs:"
        echo "  tail -f slurm/${JOB_ID}_0_log.out"
        echo "  tail -f slurm/${JOB_ID}_0_log.err"
    else
        echo "Slurm directory doesn't exist yet."
        echo "This means the job hasn't started running yet."
        echo ""
        echo "Check job status:"
        echo "  squeue -u \$USER | grep $JOB_ID"
        echo ""
        echo "Once the job starts, logs will appear in:"
        echo "  $(pwd)/slurm/${JOB_ID}_0_log.out"
        echo "  $(pwd)/slurm/${JOB_ID}_0_log.err"
    fi
else
    echo "Not in an experiment directory (no heyhi.jobid file found)"
    echo ""
    echo "To find your experiment directory:"
    echo "  1. Check exps/adhoc/ if you used --adhoc"
    echo "  2. Check exps/p/ if you didn't use --adhoc"
    echo ""
    echo "Or search for it:"
    echo "  find exps/ -name 'heyhi.jobid' -exec dirname {} \\;"
fi
