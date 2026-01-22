# Finding Slurm Logs

## How Logs Are Saved

Based on the code in `heyhi/__init__.py`:

1. **Line 362**: `submitit.SlurmExecutor(folder=exp_handle.slurm_path)` - submitit writes logs to the folder specified
2. **Line 226**: `slurm_path = exp_path / "slurm"` - logs go to `{exp_path}/slurm/`
3. **Lines 456-459**: When job is submitted, it prints:
   ```
   stdout: tail -F {exp_handle.slurm_path}/{job.job_id}_0_log.out
   stderr: tail -F {exp_handle.slurm_path}/{job.job_id}_0_log.err
   ```

## Experiment Directory Structure

- **With `--adhoc` flag**: `exps/adhoc/{date}/{exp_id}/slurm/{job_id}_0_log.out`
- **Without `--adhoc`**: `exps/p/{exp_id}/slurm/{job_id}_0_log.out`

Where `exp_id` is built from your config path and overrides.

## Finding Your Logs

### Step 1: Find your experiment directory

```bash
# List adhoc experiments (if you used --adhoc)
ls -la exps/adhoc/

# Or list permanent experiments (if you didn't use --adhoc)
ls -la exps/p/
```

### Step 2: Find the job ID

```bash
# Check the heyhi.jobid file in your experiment directory
cat exps/adhoc/{date}/{exp_id}/heyhi.jobid

# Or check current running jobs
squeue -u $USER
```

### Step 3: Find log files

```bash
# Find all slurm directories
find exps/ -type d -name "slurm"

# Find log files for a specific job ID
find exps/ -name "*8149333*" -type f

# Or navigate to the experiment directory and check slurm/
cd exps/adhoc/{date}/{exp_id}/
ls -la slurm/
```

### Step 4: View logs

```bash
# View stdout log
tail -f exps/adhoc/{date}/{exp_id}/slurm/{job_id}_0_log.out

# View stderr log
tail -f exps/adhoc/{date}/{exp_id}/slurm/{job_id}_0_log.err
```

## Quick Commands

```bash
# Find all log files for job 8149333
find exps/ -path "*/slurm/*8149333*" -type f

# Find all slurm log directories
find exps/ -type d -name "slurm" -exec ls -la {} \;

# List all experiment directories
find exps/ -name "heyhi.jobid" -exec dirname {} \;
```
