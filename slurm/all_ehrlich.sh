CMDS="${@:1}"

for METHOD in vsd-mf vsd-lstm vsd-tfm cbas-mf cbas-lstm cbas-tfm lambo2 ga
do
	for SEED in 42 666 121 11 391
	do
		sbatch run_ehrlich.sh --solver $METHOD --seed $SEED $CMDS
	done
done
