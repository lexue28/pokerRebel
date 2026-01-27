# How to Check Available Memory on SLURM Nodes

## 1. Check Partition Memory Limits

```bash
# See memory limits for mit_normal partition
sinfo -p mit_normal -o "%P %m %c %G" 

# More detailed partition info
sinfo -p mit_normal -o "%P %l %L %m %c %G %T"
```

**Output explanation:**
- `%P` = Partition name
- `%m` = Memory per node (total)
- `%c` = CPUs per node
- `%G` = Generic resources (GPUs, etc.)
- `%l` = Time limit
- `%L` = Time limit remaining
- `%T` = State (idle, allocated, etc.)

## 2. Check Specific Node Memory

```bash
# If you have a job running, check the node
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %N"

# Check specific node details
scontrol show node <node_name>

# Example: Check node1602
scontrol show node node1602
```

**Look for:**
- `RealMemory=` = Total memory on node
- `AllocMem=` = Currently allocated memory
- `FreeMem=` = Available memory

## 3. Check Memory on Currently Running Job

```bash
# Get your running job ID
squeue -u $USER

# Check memory usage of running job
scontrol show job <JOBID> | grep -E "Memory|MaxRSS"

# Or SSH to the node and check
ssh <node_name>
free -h
```

## 4. Check Memory Limits for Your Account

```bash
# Check your account limits
sacctmgr show user $USER -p

# Check QOS limits (Quality of Service)
sacctmgr show qos -p
```

## 5. Quick Memory Check Commands

```bash
# See all nodes and their memory
sinfo -o "%N %m %c %T" | head -20

# See idle nodes with memory info
sinfo -p mit_normal -t idle -o "%N %m %c %T"

# Check memory on login node (where you are now)
free -h

# Check memory on compute node (if you have access)
ssh node1602 free -h
```

## 6. Find Maximum Requestable Memory

```bash
# Check partition max memory
sinfo -p mit_normal -o "%P %m"

# Check if there are per-user limits
sacctmgr show assoc user=$USER -p | grep -i mem
```

## 7. Monitor Memory Usage During Job

```bash
# On the compute node (if you have interactive session)
watch -n 1 free -h

# Or check process memory
top -u $USER
# Press 'M' to sort by memory

# Check specific process
ps aux | grep python | awk '{print $2, $6/1024 "MB"}'
```

## 8. Check Memory After Job Completes

```bash
# Check peak memory usage from completed job
sacct -j <JOBID> --format=JobID,MaxRSS,ReqMem,AllocMem

# Example:
sacct -j 8454311 --format=JobID,MaxRSS,ReqMem,AllocMem
```

**Output explanation:**
- `MaxRSS` = Maximum memory actually used
- `ReqMem` = Memory you requested
- `AllocMem` = Memory SLURM allocated

## Recommended Workflow

**Step 1: Check partition limits**
```bash
sinfo -p mit_normal -o "%P %m %c"
```

**Step 2: Check if nodes have enough memory**
```bash
sinfo -p mit_normal -t idle -o "%N %m %T"
```

**Step 3: Request reasonable amount**
- If nodes have 256G total → request 192G-256G
- If nodes have 128G total → request 96G-128G
- Leave ~20% headroom for system overhead

**Step 4: Monitor actual usage**
```bash
# After job starts
scontrol show job <JOBID> | grep MaxRSS
```

**Step 5: Adjust for next run**
- If MaxRSS < 50% of requested → can reduce request
- If MaxRSS > 90% of requested → increase request or reduce parameters

## Typical MIT Engaging Node Memory

Most `mit_normal` nodes have:
- **128G-256G** total memory per node
- **10-20 CPUs** per node
- Can request up to node total (but leave headroom)

## Quick Check Right Now

```bash
# See what memory nodes have
sinfo -p mit_normal -o "%N %m %c %T" | head -10

# Check your current job's memory
squeue -u $USER -o "%.18i %.9P %.8j %.10M %N"
```
