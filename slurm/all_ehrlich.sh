CMDS="${@:1}"

for METHOD in vsd-lstm vsd-tfm cbas-lstm cbas-tfm ga #lambo2
do
	for SEED in 42 666 121 11 391
	do
		sbatch slurm/run_ehrlich.sh --solver $METHOD --seed $SEED $CMDS
	done
done
