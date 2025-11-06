CMDS="${@:1}"

for METHOD in lambo2 vsd-tfm vsd-mtfm cbas-tfm cbas-mtfm rand agps-tfm agps-mtfm
do
	for SEED in 42 666 121 11 391
	do
		sbatch slurm/run_ehrlichnat.sh --solver $METHOD --seed $SEED $CMDS
	done
done
