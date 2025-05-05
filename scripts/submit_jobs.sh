for v in videos/*.mp4; do
    sbatch run_hpc_job.sh "$v"
done