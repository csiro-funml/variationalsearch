CMDS="${@:1}"

for BBOX in BraninCurrin DTLZ2 DTLZ7 ZDT3 GMM DTLZ2-5
do
	sbatch slurm/run_synfn_moo.sh --bbox $BBOX $CMDS
done
