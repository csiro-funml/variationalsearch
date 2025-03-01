DATASET=$1
CMDS="${@:2}"

for METHOD in VSD CbAS DbAS BORE Random AdaLead PEX
do
	for SEED in 42 666 121 11 391
	do
		sbatch run_seq_exp.sh $DATASET $METHOD --seed $SEED $CMDS
	done
done
