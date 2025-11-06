CMDS="${@:1}"

for METHOD in vsd-mtfm cbas-mtfm agps-mtfm rand lambo2
do
	for SEED in 42 666 121 11 391
	do
		sbatch slurm/run_foldx.sh --solver $METHOD --seed $SEED $CMDS
	done
done
