# Variational search distributions

Reference implementation of [variational search
distributions](https://arxiv.org/abs/2409.06142) (VSD) for active generation,
and code for running the experiments in the paper.


## License

Copyright 2025 Commonwealth Scientific and Industrial Research Organisation
(CSIRO)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Installation

If you wish to use VSD as a library, then you can include it in your
`pyproject.toml` under `dependencies` as,

    "vsd@git+ssh://git@github.com/csiro-funml/variationalsearch.git"

Alternatively if you want to develop VSD, then basic instructions for
installation with `venv` and `pip` are,

    cd path/to/variationalsearch
    python3 -m venv .env
    source .env/bin/activate
    pip install .

Or to also run the experiments, replace the last line above with,

    pip install .[experiments]
    git lfs install
    git lfs pull

If you wish to use to
[poli](https://github.com/MachineLearningLifeScience/poli) solver interface or
to run the Ehrlich function experiment, you will need to use the `poli` optional
dependencies,

    pip install .[poli]


For all functionality and to develop, we recommend installing as below,

    cd path/to/variationalsearch
    python3 -m venv .env
    source .env/bin/activate
    pip install -e .[experiments,poli]
    git lfs install
    git lfs pull


### Development

For development, we recommend installing at least

    pip install .[dev]

Which will bring in pytest, and allow you to run the unit tests,

    pytest .


## Experiments


### Notebook

We have included a toy experiment based on finding the super-level distribution
of a 2D mixture of Gaussians. This is probably a good starting point for
understanding all of VSD's components. This demonstrates VSD's ability to also
work in continuous spaces with only modification to the variational
distributions. See `notebooks/gmm_exploration_test.ipynb`.

This experiment was not featured in the paper.


### Biological sequence experiments

These instructions are for the _fitness landscape_ and _black-box optimization_
experiments that involve real sequences (DHFR, TrpB, TFBIND8, GFP and AAV) in
Sections 4.2 and 4.3. All experimental configuration is contained in
`experiments/config.py`.

Before running the experiments, if you do not have any pre-trained predictive
models in the `models/` directory, or if you wish to modify the predictive
models, you will need to run,

    train_cpe --dataset DATASET

to train a class probability estimator, or

    train_gp --dataset DATASET

for training a Gaussian process surrogate. See `train_cpe --help` and
`train_gp --help` for all options. We give the options for `DATASET` below.

- `GFP` for the partially synthetic (oracle predictor) avGFP dataset,
- `DHFR` for the complete-space DHFR dataset,
- `TFBIND8` for the complete-space TF bind 8 datasets,
- `AAV` for the partially synthetic (oracle predictor) AAV dataset,
- `TRPB` for the complete-space TrpB dataset.

Some of these datasets also have a `_FL` prefix, e.g. `DHFR_FL`, which refers to
the "fitness landscape" version of an experiment. That is, where the fitness
threshold is fixed, and precision and recall are measured.

You may also wish to train a prior on some data, this can be done by calling the
`train_prior` command.

    train_prior --dataset DATASET

You can choose to train a prior on all the initial predictor training data, or
just the `fit` subset, see the `dataset["prior"]["use_threshold"]`
configuration parameter.

The experiments can be run by executing the shell script,

    ./run_seq_exp DATASET

Or, you can specify the dataset and method to run by calling,

    batch_rounds --dataset DATASET --method METHOD

See `batch_rounds --help` for all options and option values.

All surrogate model and experimental configuration can be viewed and changed in
`experiments/config.py`. Furthermore, an experiment will save its configuration
to a JSON file, which can also be modified and then used in the
`batch_rounds --config CONFIG` argument.

Run the following commands to replicate the results from the paper (you may
need to parallelize these on multiple GPUs). For the fitness landscape
experiments (section 4.2):

    run_seq_exp DHFR_FL
    run_seq_exp TRPB_FL
    run_seq_exp TFBIND8_FL

or to run the Gaussian process version in appendix section C.1, append the
`-gp` flag to the above command. For the black-box optimization experiments run
the following,

    run_seq_exp AAV
    run_seq_exp GFP

While the fitness landscape experiments can be run on a laptop, we run the BBO
experiments on a single NVIDIA H100 GPU equipped server. After these scripts
have completed, run the following command to create the plots we present in the
paper,

    plot_results --datadir DATASET

where `DATASET` is a dataset listed above. To run the ablation experiments, you
will need to modify the configuration file `AAV` or `GFP` configuration
dictionaries as we have commented in `experiments/config.py` and the run the
`batch_rounds` command explicitly for `--method VSD`.

We have also provided a [slurm](https://slurm.schedmd.com/sacct.html) run script
for these experiments, see `slurm/run_seq_exp.sh` and `slurm/all_seq_exp.sh`.


### Ehrlich functions

The Ehrlich functions operate slightly differently from the biological
sequences since they are from the
[poli](https://github.com/MachineLearningLifeScience/poli) benchmarks library.

They have their own run script in `experiments/scripts/ehrlich.py` which can be
run from the command line as,

    ehrlich

or run `ehrlich --help` to see a full list of options. For example, to run VSD
with a transformer on the Ehrlich functions with a sequence length of 15 you
could run,

    ehrlich --solver vsd-tfm --sequence-length 15 --max-iter 32 --seed 42

This will log all results to the `ehrlich` directory by default, or you can
specify another `--logdir` location. To plot results, run

    plot_ehrlich --resultsdir ehrlich

Or wherever you decide to save results. We also provide some convenience
scripts for running on a [slurm](https://slurm.schedmd.com/sacct.html) based
cluster, see `slurm/run_ehlich.sh` and `slurm/all_ehrlich.sh`.


### Handwritten digit generation

The handwritten digit generation experiment (section 4.1 from the paper) is
self-contained in the `experiments/scripts/digits.py` script, to run simply call

    digits

for running the digits experiments with the transformer variational
distribution, or append the `-lstm` flag to use the LSTM variational
distribution. NOTE: this will require a decent GPU with 32GB+ of memory to run
without modification -- we use an NVIDIA H100 GPU. All configuration for this
experiment is contained within the script.

We have also provided a [slurm](https://slurm.schedmd.com/sacct.html) run script
for these experiments, see `slurm/run_digits.sh`.


## Citation

Please cite us if you use this work:

_Daniel M. Steinberg, Rafael Oliveira, Cheng Soon Ong, Edwin V. Bonilla.
The Thirteenth International Conference on Learning Representations (ICLR), 2025._

```bibtex
@inproceedings{steinberg2025variational,
  title={Variational Search Distributions},
  author={Steinberg, Daniel M and Oliveira, Rafael and Ong, Cheng Soon and Bonilla, Edwin V},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```


## Acknowledgements
- TFBIND8 data, https://huggingface.co/datasets/beckhamc/design_bench_data
- GFP and AAV data, https://github.com/kirjner/GGS
- DHFR data, supplementary material from
  https://doi.org/10.1126/science.adh3860
- TrpB data, supplementary material from https://doi.org/10.22002/h5rah-5z170.
- Ehrlich functions, [poli](https://github.com/MachineLearningLifeScience/poli)
  benchmarks.
