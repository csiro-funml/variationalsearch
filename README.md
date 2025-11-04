# Variational search distributions

Reference implementations of:

[Variational search distributions](https://arxiv.org/abs/2409.06142) (VSD) for
active generation.

[Amortized Active Generation of Pareto Sets](https://arxiv.org/pdf/2510.21052)
(A-GPS) for multiple-objective generation.

And all code for running the experiments in the papers.


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

### Installing the VSD experiments

To run the exact experiments and algorithms from the ICLR 2025 _Variational
Search Distributions_ paper, firstly check out the `ICLR` tag,

    git clone git@github.com:csiro-funml/variationalsearch.git
    git checkout ICLR

Then install,

    cd variationalsearch
    python3 -m venv .env
    source .env/bin/activate
    pip install .[experiments,poli]
    git lfs install
    git lfs pull

The `poli` dependency will also bring in the
[poli](https://github.com/MachineLearningLifeScience/poli) library for running
the Ehrlich experiments.

Git Large File Store (LFS) is required to pull down the model weights and
datasets, see the [instructions](https://git-lfs.com) for details on installing
Git LFS.

These experiments will also run on the latest `main` branch, however the
methods may include improvements (such as the off-policy gradient estimator)
from the A-GPS paper.

### Installing the A-GPS experiments

To run the exact experiments from the NeurIPS 2025 _Amortized Active Generation
of Pareto Sets_ paper, simply check out the latest version of `main` and
install,

    cd variationalsearch
    python3 -m venv .env
    source .env/bin/activate
    pip install .[experiments,poli-moo]
    git lfs install
    git lfs pull

The `poli-moo` dependency is required here since we need the `foldx` black box
model from poli, and we had to modify the poli-baselines version of LaMBO-2 to
expose the EHVI acquisition functions.

### Development

For development, we recommend installing at least

    pip install .[dev]

Which will bring in pytest, and allow you to run the unit tests,

    pytest .


## Usage

VSD and A-GPS are highly modular algorithms. The fastest way to start is to use
the interfaces in `solvers.py` that are compatible with the poli library.

You can also look at the `notebooks/gmm_exploration_test.ipynb` notebook for
a simple implementation of VSD. And `experiments/scripts/synfn_moo.py` for a
simple implementation of A-GPS. However, both of these use simple continuous
generative models.

Otherwise, look at the following files in the `vsd` package for the various
components.

- `generation`: Optimisation routines for candidate generation (BoTorch style),
    this implements reinforce, off-policy optimisation, estimation of
    distribution algorithms (weighted MLE) and Adalead and PEX. This is a good
    place to start for piecing things together.
- `acquisition`: Acquisition functions, and "meta" acquisition functions that
    implements VSD's ELBO and A-GPS's A-ELBO losses.
- `cpe`: Class probability estimators, p(z|x) and p(z|x, u).
- `surrogates`: Gaussian process surrogate models (e.g. for GP-PI).
- `proposals`: Generative models and priors (unconditional), q(x).
- `condproposals`: Conditional generative models (for A-GPS), and a "shim" class
    for adapting preference distributions and conditional proposals for use with
    the functions in `generation` (`PreferenceSearchDistribution`), q(x|u).
- `labellers`: Thresholding/Non-dominance labelling functions, for constant and
    adaptive labelling methods.
- `preferences`: Preference direction vector distributions, q(u|z).


## Experiments

If you wish to replicate the VSD experiments using the original published
method, please follow the _Install the VSD experiments_ instructions. These
experiments will still work otherwise with the more recent (A-GPS) codebase,
they may just include some improvements from the later work.

### Notebook

We have included a toy experiment based on finding the super-level distribution
of a 2D mixture of Gaussians. This is probably a good starting point for
understanding all of VSD's components. This demonstrates VSD's ability to also
work in continuous spaces with only modification to the variational
distributions. See `notebooks/gmm_exploration_test.ipynb`.

This experiment was not featured in the paper.


### VSD Biological sequence experiments

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

Some of these datasets also have a `_FL` prefix, e.g. `DHFR_FL`, which refers
to the "fitness landscape" version of an experiment. That is, where the fitness
threshold is fixed, and precision and recall are measured.

You may also wish to train a prior on some data, this can be done by calling
the `train_prior` command.

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


### VSD Ehrlich functions

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


### VSD Handwritten digit generation

The handwritten digit generation experiment (section 4.1 from the paper) is
self-contained in the `experiments/scripts/digits.py` script, to run simply call

    digits

for running the digits experiments with the transformer variational
distribution, or append the `-lstm` flag to use the LSTM variational
distribution. NOTE: this will require a decent GPU with 32GB+ of memory to run
without modification -- we use an NVIDIA H100 GPU. All configuration for this
experiment is contained within the script.

We have also provided a [slurm](https://slurm.schedmd.com/sacct.html) run
script for these experiments, see `slurm/run_digits.sh`.

### A-GPS Synthetic Functions

The synthetic multi-objective benchmarks from the paper are provided via the
`synfn_moo` entry point (`experiments/scripts/synfn_moo.py`). This runner
initialises a Latin hypercube of size `--tsize` and then compares A-GPS against
the qNParEGO, qEHVI and qNEHVI baselines whenever the number of objectives
allows it. For example,

    synfn_moo --bbox BraninCurrin --max-iter 10 --replicates 10 --bsize 5 --tsize 64 --preference gmm --logdir runs/synfn

Choose `--bbox` from `BraninCurrin`, `DTLZ2`, `DTLZ2-4`, `DTLZ7`, `ZDT3` or
`GMM`. Each problem writes to `<logdir>/<bbox>/`, producing a run log, a
hypervolume trace (`*_hypervol.png`) and a NumPy archive (`results.npz`) with
the round-by-round hypervolume curves for every baseline. The `--replicates`
flag repeats the full loop with fresh seeds, while `--preference` selects the
A-GPS preference prior (`gmm`, `normal` or `empirical`). Slurm templates for
sweeping all benchmarks live in `slurm/run_synfn_moo.sh` and
`slurm/all_synfn_moo.sh`.

### A-GPS Ehrlich vs. Naturalness

The Ehrlich versus naturalness study couples the poli Ehrlich landscape with
the ProtBERT naturalness score and is implemented by the `ehrlich_nat` entry
point (`experiments/scripts/ehrlich_nat.py`). Installing the `[poli-moo]` extra
ensures all dependencies are available. A typical A-GPS run is

    ehrlich_nat --solver agps-tfm --sequence-length 32 --max-iter 40 --bsize 32 --tsize 128 --device cuda --logdir runs/ehrlich_nat

Set `--solver` to one of the A-GPS, VSD, CbAS, LaMBO-2 or random mutation
baselines enumerated in the script (transformer, LSTM and mutation-conditioned
variants, with optional reinforcement fine-tuning denoted by the `-rf` suffix).
The `--sequence-length` flag accepts 15, 32 or 64 and picks the motif and
mutation settings described in the paper. Run logs, Pareto/hypervolume plots
and the raw sequences plus evaluations (`*.npz`) are saved in
`<logdir>/<sequence-length>/<solver>_<seed>.*`. When using A-GPS an additional
`*_samples.png`/`.npz` pair stores preference-conditioned samples. Use `--holo`
to switch to the original holo implementation of the Ehrlich benchmark, `--ref`
to override the hypervolume reference point and `--gsamples` to control
gradient Monte Carlo samples. Slurm wrappers for batch execution are provided
in `slurm/run_ehrlichnat.sh` and `slurm/all_ehrlichnat.sh`, and aggregate plots
can be produced with `plot_moo_2d --resultsdir runs/ehrlich_nat/32 --ref -1 0`.

### A-GPS Bi-/Ngrams

The bi- and n-gram regex optimisation tasks use the `ngrams` entry point
(`experiments/scripts/ngrams.py`), which constructs the tri-objective
regex problem described in the paper. The script seeds the buffer with
`--tsize` examples and supports A-GPS, VSD, CbAS, LaMBO-2 and random
mutation baselines. For example,

    ngrams --solver agps-mtfm --max-iter 64 --bsize 16 --tsize 512 --device cuda --logdir runs/ngrams

Results are written to `<logdir>/<solver>_<seed>.*`, including a log,
summary plots of the cumulative objective counts and a `*.npz` archive
with every sequence/evaluation pair. Use `plot_ngrams --resultsdir runs/ngrams`
to aggregate hypervolume and diversity statistics across seeds. Slurm
templates for cluster execution are available in `slurm/run_ngrams.sh`
and `slurm/all_ngrams.sh`.

### A-GPS Stability vs. SASA

The FoldX stability versus SASA experiment is driven by the `foldx_ss` entry
point (`experiments/scripts/foldx_ss.py`). Install the FoldX binary and ensure
`poli-core` can discover it (follow the FoldX guidance linked above), then
place the RFP PDB files from LaMBO under `data/rfp_pdbs` (should be included)
or supply the directory via `--datapath`. On the first run the script rebuilds
`rfp_training_cache.csv` using your local FoldX version, so expect an initial
warm-up pass. A representative command is

    foldx_ss --solver agps-mtfm --max-iter 64 --bsize 16 --tsize 512 --device cuda --datapath data/rfp_pdbs --logdir runs/foldx

Choose `--solver` from the A-GPS, VSD, CbAS, LaMBO-2 or random pad-mutation
options. Outputs follow the same convention as above:
`<logdir>/<solver>_<seed>.*` holds the log file, Pareto/hypervolume plot, and a
`*.npz` file with the sequences (padded to a common length) and scores. A-GPS
runs also emit preference-conditioned samples (`*_samples.*`). Adjust `--ref`
to set the hypervolume reference point, the default is inferred from the data
(experiments inferred the reference points). Slurm helpers live in
`slurm/run_foldx.sh` and `slurm/all_foldx.sh`, and aggregate plots can be
generated with `plot_moo_2d --resultsdir runs/foldx`, and you will need to
manually set the reference point, `--ref f1 f2` for `f1` and `f2` here, we used
-90 10000 for the A-GPS paper.

## Citations

Please cite us if you use this work:

Daniel M. Steinberg, Rafael Oliveira, Cheng Soon Ong, Edwin V. Bonilla.
_Variational Search Distributions_.
The Thirteenth International Conference on Learning Representations (ICLR), 2025.

```bibtex
@inproceedings{steinberg2025variational,
  title={Variational Search Distributions},
  author={Steinberg, Daniel M and Oliveira, Rafael and Ong, Cheng Soon
    and Bonilla, Edwin V},
  booktitle={The Thirteenth International Conference on Learning
    Representations (ICLR)},
  year={2025}
}
```

Or,

Daniel M. Steinberg, Asiri Wijesinghe, Rafael Oliveira, Piotr Koniusz,
Cheng Soon Ong, Edwin V. Bonilla.
_Amortized Active Generation of Pareto Sets_.
Neural Information Processing Systems (NeurIPS), 2025.

```bibtex
@inproceedings{steinberg2025amortized,
  title={Amortized Active Generation of Pareto Sets},
  author={Steinberg, Daniel M and Wijesinghe, Asiri and Oliveira, Rafael and
    Koniusz, Piotr and Ong, Cheng Soon and Bonilla, Edwin V},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Acknowledgements
- TFBIND8 data, https://huggingface.co/datasets/beckhamc/design_bench_data
- GFP and AAV data, https://github.com/kirjner/GGS
- DHFR data, supplementary material from
  https://doi.org/10.1126/science.adh3860
- TrpB data, supplementary material from https://doi.org/10.22002/h5rah-5z170.
- Ehrlich functions and Foldx black boxes,
    [poli](https://github.com/MachineLearningLifeScience/poli) benchmarks.
- LaMBO for the RFP data and original Foldx experiment https://github.com/samuelstanton/lambo.
