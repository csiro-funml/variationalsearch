CMDS="${@:1}"

for METHOD in vsd-tfm vsd-mtfm cbas-tfm cbas-mtfm agps-tfm agps-mtfm rand lambo2
do
	for SEED in 42 666 121 11 391
	do
		sbatch slurm/run_ngrams.sh --solver $METHOD --seed $SEED $CMDS
	done
done
