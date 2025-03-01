"""Experimental configuration."""

from copy import deepcopy

#
# Global configs
#


# Model cache
MODEL_PATH = "models"


# Data locations
DATA_DIR = "data"


# General data properties
NUCLEIC_ALPHA = list("ACGT")
AMINO_ALPHA = list("ARNDCQEGHILKMFPSTWYV")


#
# Biological Experiment configs
#


DHFR_DATA = {
    "data": "DHFR",
    "seed": 666,
    "t_rounds": 10,
    "b_cands": 128,
    # "threshold": {
    #     "class": "MaxIncumbent",
    #     "args": dict(eps=1e-3),
    # },
    "threshold": {
        "class": "AnnealedThreshold",
        "args": dict(percentile=0.5, eta=0.55),
    },
    "device": "cpu",
    "data_source": DATA_DIR + "/DHFR/DHFR_fitness_data_wt.csv",
    "mutation_column": "SV",
    "target_column": "m",
    "training_data": {
        "save_path": DATA_DIR + "/DHFR/surrogate_training_data.csv",
        "include_scan": False,
        "size": 2000,
        "max_fitness": -0.5,
    },
    "prior": {
        "class": "SequenceUninformativePrior",
        "parameters": {},
        "trainable": False,
    },
    "proposal": {
        "class": "MultiCategoricalProposal",
        "parameters": {},
        "from_prior": False,
        "samples": 1024,
        "optimisation": {"lr": 1e-3, "weight_decay": 0.0},
        "stop": {"maxiter": 20000, "n_window": 5000},
        "pex_options": dict(max_mutations=5),
        "adalead_options": dict(max_mutations=5, kappa_cutoff=0.5),
    },
    "cpe": {
        "path": MODEL_PATH + "/DHFR_surrogate_nn.pt",
        "class": "NNClassProbability",
        "parameters": dict(hlsize=32, dropoutp=0.2, embedding_dim=8),
        "batchsize": 512,
        "optimisation": {"lr": 1e-3},
        "stop": {"maxiter": 20000, "n_window": 4000, "eta": 0.2},
    },
    "gp": {
        "path": MODEL_PATH + "/DHFR_surrogate_gp.pt",
        "class": "CategoricalGP",
        "parameters": dict(ard=True),
        "optimisation": {},
        "stop": {},
    },
}


# DHFR Fitness landscape experiment
DHFR_FL_DATA = deepcopy(DHFR_DATA)
DHFR_FL_DATA.update(
    {
        "data": "DHFR_FL",
        "threshold": {"class": "Threshold", "args": dict(best_f=-0.1)},
        "training_data": {
            "save_path": DATA_DIR + "/DHFR/FL_surrogate_training_data.csv",
            "include_scan": False,
            "include_wt": True,
            "size": 2000,
            "max_fitness": 0.01,
        },
    }
)
DHFR_FL_DATA["cpe"]["path"] = MODEL_PATH + "/DHFR_FL_surrogate_nn.pt"
DHFR_FL_DATA["gp"]["path"] = MODEL_PATH + "/DHFR_FL_surrogate_gp.pt"


TRPB_DATA = {
    "data": "TRPB",
    "seed": 666,
    "t_rounds": 10,
    "b_cands": 128,
    "threshold": {
        "class": "AnnealedThreshold",
        "args": dict(percentile=0.5, eta=0.55),
    },
    "device": "cpu",
    "data_source": DATA_DIR + "/TRPB/four-site_simplified_AA_data.csv",
    "mutation_column": "AAs",
    "target_column": "fitness",
    "training_data": {
        "save_path": DATA_DIR + "/TRPB/surrogate_training_data.csv",
        "include_scan": False,
        "size": 2000,
        "max_fitness": 0.055,  # minimum active
    },
    "prior": {
        "class": "SequenceUninformativePrior",
        "parameters": {},
        "trainable": False,
    },
    "proposal": {
        "class": "MultiCategoricalProposal",
        "parameters": {},
        "from_prior": False,
        "samples": 1024,
        "optimisation": {"lr": 1e-3, "weight_decay": 0.0},
        "stop": {"maxiter": 20000, "n_window": 5000},
        "pex_options": dict(max_mutations=5),
        "adalead_options": dict(max_mutations=5, kappa_cutoff=0.5),
    },
    "gp": {
        "path": MODEL_PATH + "/TRPB_surrogate_gp.pt",
        "class": "CategoricalGP",
        "parameters": dict(ard=True),
        "optimisation": {},
        "stop": {},
    },
    "cpe": {
        "path": MODEL_PATH + "/TRPB_surrogate_nn.pt",
        "class": "NNClassProbability",
        "parameters": dict(hlsize=32, dropoutp=0.2, embedding_dim=8),
        "batchsize": 512,
        "optimisation": {"lr": 1e-3},
        "stop": {"maxiter": 20000, "n_window": 4000, "eta": 0.2},
    },
}


# TRPB Fitness landscape experiment
TRPB_FL_DATA = deepcopy(TRPB_DATA)
TRPB_FL_DATA.update(
    {
        "data": "TRPB_FL",
        "threshold": {"class": "Threshold", "args": dict(best_f=0.35)},
        "training_data": {
            "save_path": DATA_DIR + "/TRPB/FL_surrogate_training_data.csv",
            "include_scan": False,
            "include_wt": True,
            "size": 2000,
            "max_fitness": 0.409,
        },
    }
)
TRPB_FL_DATA["cpe"]["path"] = MODEL_PATH + "/TRPB_FL_surrogate_nn.pt"
TRPB_FL_DATA["gp"]["path"] = MODEL_PATH + "/TRPB_FL_surrogate_gp.pt"


TFBIND8_DATA = {
    "data": "TFBIND8",
    "seed": 666,
    "t_rounds": 10,
    "b_cands": 128,
    "threshold": {
        "class": "QuantileThreshold",
        "args": dict(percentile=0.95),
    },
    "device": "cpu",
    "data_source": DATA_DIR + "/TFBIND8/tf_bind_8.csv",
    "mutation_column": "sequences",
    "target_column": "fitness",
    "training_data": {
        "save_path": DATA_DIR + "/TFBIND8/surrogate_training_data.csv",
        "include_scan": False,
        "size": 2000,
        "max_fitness": 0.3,
    },
    "prior": {
        "class": "SequenceUninformativePrior",
        "trainable": False,
        "parameters": {},
    },
    "proposal": {
        "class": "MultiCategoricalProposal",
        "parameters": {},
        "from_prior": False,
        "samples": 1024,
        "optimisation": {"lr": 1e-3, "weight_decay": 0.0},
        "stop": {"maxiter": 20000, "n_window": 5000},
        "pex_options": dict(max_mutations=5),
        "adalead_options": dict(max_mutations=5, kappa_cutoff=0.5),
    },
    "gp": {
        "path": MODEL_PATH + "/TFBIND8_surrogate_gp.pt",
        "class": "CategoricalGP",
        "parameters": dict(ard=True),
        "optimisation": {},
        "stop": {},
    },
    "cpe": {
        "path": MODEL_PATH + "/TFBIND8_surrogate_nn.pt",
        "class": "NNClassProbability",
        "parameters": dict(hlsize=32, dropoutp=0.2, embedding_dim=8),
        "batchsize": 512,
        "optimisation": {"lr": 1e-3},
        "stop": {"maxiter": 20000, "n_window": 4000, "eta": 0.2},
    },
}

# TFBIND8 Fitness landscape experiment
TFBIND8_FL_DATA = deepcopy(TFBIND8_DATA)
TFBIND8_FL_DATA.update(
    {
        "data": "TFBIND8_FL",
        "threshold": {"class": "Threshold", "args": dict(best_f=0.75)},
        "training_data": {
            "save_path": DATA_DIR + "/TFBIND8/FL_surrogate_training_data.csv",
            "include_scan": False,
            "size": 2000,
            "max_fitness": 0.85,
        },
    }
)
TFBIND8_DATA["cpe"]["path"] = MODEL_PATH + "/TFBIND8_FL_surrogate_nn.pt"
TFBIND8_DATA["gp"]["path"] = MODEL_PATH + "/TFBIND8_FL_surrogate_gp.pt"


GFP_DATA = {
    "data": "GFP",
    "seed": 666,
    "t_rounds": 10,
    "b_cands": 128,
    "threshold": {
        "class": "AnnealedThreshold",
        "args": dict(percentile=0.8, eta=0.7),
    },
    "device": "mps",
    "data_source": DATA_DIR + "/GFP/ground_truth.csv",
    "mutation_column": "sequence",
    "target_column": "target",
    "oracle_path": MODEL_PATH + "/gfp_cnn_oracle.pt",
    "training_data": {
        "save_path": DATA_DIR + "/GFP/surrogate_training_data.csv",
        "oracle_targets": False,
        "size": 2000,
        "max_fitness": 1.9,
        "min_fitness": 1.31,
    },
    "prior": {
        # --- VSD-IU ablation
        # "class": "SequenceUninformativePrior",
        # "parameters": {},
        # "trainable": False,
        # --- VSD-I, VSD-TCNN and VSD-TAE methods
        # "save_path": MODEL_PATH + "/GFP_prior.pt",
        # "class": "MultiCategoricalProposal",
        # "parameters": {},
        # "trainable": True,
        # --- VSD-DTFM method
        "save_path": MODEL_PATH + "/GFP_prior_ar.pt",
        "class": "DTransformerProposal",
        "parameters": dict(
            nhead=4, num_layers=1, dim_feedforward=64, clip_gradients=1.0
        ),
        "trainable": True,
        # --- VSD-LSTM method
        # "save_path": MODEL_PATH + "/GFP_prior_lstm.pt",
        # "class": "LSTMProposal",
        # "parameters": dict(hidden_size=32, num_layers=4, clip_gradients=1.0),
        # "trainable": True,
        "batchsize": 512,
        "optimisation": {"lr": 1e-3, "weight_decay": 1e-4},
        "stop": {"maxiter": 20000, "n_window": 1000},
        "use_threshold": False,
    },
    "proposal": {
        # --- VSD-IU and VSD-I methods
        # "class": "MultiCategoricalProposal",
        # "parameters": {},
        # "optimisation": {"lr": 1e-3},
        # "from_prior": False, # For VSD-IU
        # "from_prior": True, # For VSD-I
        # --- VSD-TCNN method
        # "class": "TransitionCNNProposal",
        # "parameters": dict(
        #     latent_k=64,
        #     kernel_size=7,
        # ),
        # "optimisation": {"lr": 1e-3},
        # "from_prior": False,
        # --- VSD-TAE method
        # "class": "TransitionAEProposal",
        # "parameters": dict(
        #     embedding_dim=16,
        # ),
        # "optimisation": {"lr": 1e-3},
        # "from_prior": False,
        # --- VSD-DTFM method
        "class": "DTransformerProposal",
        "parameters": dict(
            nhead=4, num_layers=1, dim_feedforward=64, clip_gradients=1.0
        ),
        "optimisation": {"lr": 1e-4, "weight_decay": 0.0},
        "from_prior": True,
        # --- VSD-LSTM method
        # "class": "LSTMProposal",
        # "parameters": dict(hidden_size=32, num_layers=4, clip_gradients=1.0),
        # "optimisation": {"lr": 1e-4},
        # "from_prior": True,
        "samples": 512,
        "stop": {"maxiter": 40000, "n_window": 5000},
        "pex_options": dict(max_mutations=10),
        "adalead_options": dict(max_mutations=10, kappa_cutoff=0.5),
    },
    "gp": {
        "path": MODEL_PATH + "/GFP_surrogate_gp.pt",
        "class": "CategoricalGP",
        "parameters": dict(ard=True),
        "optimisation": {},
        "stop": {},
    },
    "cpe": {
        "path": MODEL_PATH + "/GFP_surrogate_cnn.pt",
        "class": "CNNClassProbability",
        "parameters": dict(
            ckernel=7,
            xkernel=4,
            xstride=4,
            cfilter_size=16,
            linear_size=128,
            dropoutp=0.2,
        ),
        "batchsize": 512,
        "optimisation": {"lr": 1e-3, "weight_decay": 0.0},
        "stop": {"maxiter": 10000, "n_window": 4000, "eta": 0.2},
    },
}


AAV_DATA = {
    "data": "AAV",
    "seed": 666,
    "t_rounds": 10,
    "b_cands": 128,
    "threshold": {
        "class": "AnnealedThreshold",
        "args": dict(percentile=0.8, eta=0.7),
    },
    "device": "mps",
    "data_source": DATA_DIR + "/AAV/ground_truth.csv",
    "mutation_column": "sequence",
    "target_column": "target",
    "oracle_path": MODEL_PATH + "/aav_cnn_oracle.pt",
    "training_data": {
        "save_path": DATA_DIR + "/AAV/surrogate_training_data.csv",
        "oracle_targets": False,
        "size": 2000,
        "max_fitness": 5,
        "min_fitness": 0,
    },
    "prior": {
        # --- VSD-IU ablation
        # "class": "SequenceUninformativePrior",
        # "parameters": {},
        # "trainable": False,
        # --- VSD-I, VSD-TAE and VSD-TCNN methods
        # "save_path": MODEL_PATH + "/AAV_prior.pt",
        # "class": "MultiCategoricalProposal",
        # "parameters": {},
        # "trainable": True,
        # --- VSD-DTFM method
        "save_path": MODEL_PATH + "/AAV_prior_ar.pt",
        "class": "DTransformerProposal",
        "parameters": dict(
            nhead=2, num_layers=1, dim_feedforward=64, clip_gradients=1.0
        ),
        "trainable": True,
        # --- VSD-LSTM method
        # "save_path": MODEL_PATH + "/AAV_prior_lstm.pt",
        # "class": "LSTMProposal",
        # "parameters": dict(hidden_size=32, num_layers=4, clip_gradients=1.0),
        # "trainable": True,
        "batchsize": 512,
        "optimisation": {"lr": 1e-3, "weight_decay": 1e-4},
        "stop": {"maxiter": 20000, "n_window": 1000},
        "use_threshold": False,
    },
    "proposal": {
        # --- VSD-I and VSD-IU methods
        # "class": "MultiCategoricalProposal",
        # "parameters": {},
        # "optimisation": {"lr": 1e-3},
        # "from_prior": False,  # For VSD-IU
        # "from_prior": True,  # For VSD-I
        # --- VSD-TCNN method
        # "class": "TransitionCNNProposal",
        # "parameters": dict(
        #     latent_k=64,
        #     kernel_size=7,
        # ),
        # "optimisation": {"lr": 1e-3},
        # "from_prior": False,
        # --- VSD-TAE method
        # "class": "TransitionAEProposal",
        # "parameters": dict(
        #     embedding_dim=16,
        # ),
        # "optimisation": {"lr": 1e-3},
        # "from_prior": False,
        # --- VSD-DTFM method
        "class": "DTransformerProposal",
        "parameters": dict(
            nhead=2, num_layers=1, dim_feedforward=64, clip_gradients=1.0
        ),
        "optimisation": {"lr": 1e-4, "weight_decay": 0.0},
        "from_prior": True,
        # --- VSD-LSTM method
        # "class": "LSTMProposal",
        # "parameters": dict(hidden_size=32, num_layers=4, clip_gradients=1.0),
        # "optimisation": {"lr": 1e-4},
        # "from_prior": True,
        "samples": 512,
        "stop": {"maxiter": 40000, "n_window": 5000},
        "pex_options": dict(max_mutations=10),
        "adalead_options": dict(max_mutations=10, kappa_cutoff=0.5),
    },
    "gp": {
        "path": MODEL_PATH + "/AAV_surrogate_gp.pt",
        "class": "CategoricalGP",
        "parameters": {},
        "optimisation": {},
        "stop": {},
    },
    "cpe": {
        "path": MODEL_PATH + "/AAV_surrogate_cnn.pt",
        "class": "CNNClassProbability",
        "parameters": dict(
            ckernel=7,
            xkernel=2,
            xstride=2,
            cfilter_size=16,
            linear_size=128,
            dropoutp=0.2,
        ),
        "batchsize": 500,
        "optimisation": {"lr": 1e-3},
        "stop": {"maxiter": 10000, "n_window": 4000, "eta": 0.2},
    },
}
