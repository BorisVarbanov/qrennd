# %%
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
EXP_NAME = "training_logs"
MODEL_FOLDER: str = "20230206-114805-base_training_no-dropout_no-l2"

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()  # define the path where the notebook is placed.

# experiment folder
EXP_DIR = NOTEBOOK_DIR / "data" / EXP_NAME
if not EXP_DIR.exists():
    raise ValueError("Experiment directory does not exist.")

LOG_FILE = EXP_DIR / f"{MODEL_FOLDER}/logs/training.log"
if not LOG_FILE.exists():
    raise ValueError(f"Log file does not exist: {MODEL_FOLDER}/logs/training.log")

# %%
dataframe = pd.read_csv(LOG_FILE)

# %%
dataframe

# %%
METRICS = ("loss", "main_output_accuracy")
EPOCH_CUT = 100
acc_MWPM = 0.72482

for metric in METRICS:
    fig, axs = plt.subplots(figsize=(10, 4), ncols=2)

    axs[0].plot(
        dataframe.epoch, dataframe[metric], ".-", color="blue", label="Training"
    )
    axs[0].plot(
        dataframe.epoch,
        dataframe["val_" + metric],
        ".-",
        color="orange",
        label="Validation",
    )

    axs[1].plot(
        dataframe.epoch[EPOCH_CUT:],
        dataframe[metric][EPOCH_CUT:],
        ".-",
        color="blue",
        label="Training",
    )
    axs[1].plot(
        dataframe.epoch[EPOCH_CUT:],
        dataframe["val_" + metric][EPOCH_CUT:],
        ".-",
        color="orange",
        label="Validation",
    )

    if metric == "main_output_accuracy":
        axs[0].axhline(y=acc_MWPM, linestyle="--", color="gray", label="MWPM (test)")

    axs[0].legend(frameon=False)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel(metric.replace("_", " ").capitalize())
    axs[1].legend(frameon=False)
    axs[1].set_xlabel("Epochs")
    axs[1].set_xlim(EPOCH_CUT, max(dataframe.epoch) + 1)

plt.show()

# %% [markdown]
# # Evaluation

# %%
import xarray as xr
from qec_util import Layout
from qrennd import get_model, Config
from qec_util.util.syndrome import get_syndromes, get_defects
from qec_util.util.analysis import logical_fidelity, LogicalFidelityDecay, lmfit_par_to_ufloat


# %%
def evaluate_model(
    model,
    DIR,
    SUFFIX_DATASETS = "",
    QEC_CYCLES = list(range(1, 40)),
    LOG_STATES = [0, 1]
):

    model.load_weights(DIR / "checkpoint/weights.hdf5")
    
    LOG_FID = np.zeros((len(QEC_CYCLES), len(LOG_STATES)))
    for r_idx, r in enumerate(QEC_CYCLES):
        print(r)
        for l_idx, log_state in enumerate(LOG_STATES):
            circ = f"surf-code_d3_bZ_s{log_state}_n20000_r{r}{SUFFIX_DATASETS}"

            test_dataset = xr.load_dataset(
                EXP_DIR / "test" / circ / "measurements.nc"
            )

            test_dataset = test_dataset.rename(shot="run")
            test_input, test_output = preprocess_data(test_dataset, data_input=DATA_INPUT)
            eval_output = model.predict(x=test_input)
            p1, p2 = eval_output
            # p = p1*(1-p2) + p2*(1-p1)
            p = p1
            out = p > 0.5
            out = out.flatten()
            error_prob = np.average(out ^ test_output)
            LOG_FID[r_idx, l_idx] = 1 - error_prob 
            
    fid_arr = xr.DataArray(LOG_FID, dims=["qec_round", "log_state"], 
                           coords=dict(log_state=LOG_STATES, qec_round=list(QEC_CYCLES)))
    log_fid = fid_arr.mean(dim="log_state")
    log_fid = log_fid.to_dataset(name="log_fid")
    log_fid.to_netcdf(path=DIR / "test_results.nc")
    
    return

def preprocess_data(dataset, data_input="defects"):
    anc_meas = dataset.anc_meas.transpose("run", "qec_round", "anc_qubit")
    data_meas = dataset.data_meas.transpose("run", "data_qubit")

    syndromes = get_syndromes(anc_meas, meas_reset=dataset.meas_reset.values)
    defects = get_defects(syndromes)

    log_meas = dataset.data_meas.sum(dim="data_qubit") % 2
    log_errors = log_meas ^ dataset.log_state

    if data_input == "defects":
        inputs = dict(defects=defects.data, final_defects=data_meas.data)
    elif data_input == "syndromes":
        inputs = dict(defects=syndromes.data, final_defects=data_meas.data)
    elif data_input == "measurements":
        inputs = dict(defects=anc_meas.data, final_defects=data_meas.data)
    else:
        raise ValueError("'data_input' type must be: 'defects', 'syndromes' or 'measurements'")
    outputs = log_errors.data

    return inputs, outputs



def init_model(SETUP):
    with open(SETUP, "r") as file:
        setup = file.read()
        
    setup = setup.split("\n")
    for line in setup:
        if "CONFIG = Config(" in line:
            config = eval(line[len("CONFIG = "):])
        if "FINAL_DATA_INPUT" in line:
            if "measurements" in line:
                num_data = 9
            elif "defects" in line:
                num_data = 4
    
    num_rounds = None 
    num_anc = 8
    
    model = get_model(
        defects_shape=(num_rounds, num_anc),
        final_defects_shape=(num_data, ),
        config=config,
        metrics={},
    )
    
    return model


# %%
SETUP = EXP_DIR / f"{MODEL_FOLDER}/logs/setup.txt"
if not SETUP.exists():
    raise ValueError(f"Log file does not exist: {SETUP}")

# %%
DATA_INPUT : str = "defects" # it should be loaded from the setup file

# %%
# if results have not been stored, evaluate model
DIR = EXP_DIR / f"{MODEL_FOLDER}"
if not (DIR / "test_results.nc").exists():
    print("Evaluating model...")
    model = init_model(SETUP)
    evaluate_model(
        model,
        DIR,
    )
    
log_fid = xr.load_dataset(DIR / "test_results.nc")

# %%
MPWM_log_fid = np.array([0.96275, 0.9199 , 0.88245, 0.84325, 0.81565, 0.78715, 0.7602 ,
       0.7285 , 0.70985, 0.6915 , 0.6706 , 0.65755, 0.64445, 0.62755,
       0.62095, 0.6042 , 0.5971 , 0.5835 , 0.5842 , 0.56485, 0.5673 ,
       0.56225, 0.5579 , 0.54265, 0.53465, 0.5373 , 0.542  , 0.5274 ,
       0.52815, 0.533  , 0.52485, 0.52125, 0.5295 , 0.5174 , 0.5197 ,
       0.51355, 0.51115, 0.51   , 0.5112 , 0.5111 , 0.5084 , 0.51125,
       0.50955, 0.50385, 0.505  , 0.5107 , 0.50995, 0.5012 , 0.5022 ,
       0.50445, 0.50205, 0.50035, 0.50045, 0.49915, 0.5033 , 0.5029 ,
       0.50055, 0.4986 , 0.50015])
MWPM_qec_round = np.arange(1, 60)

# %%
model_decay = LogicalFidelityDecay()
params = model_decay.guess(log_fid.log_fid.values, x=log_fid.qec_round.values)
out = model_decay.fit(log_fid.log_fid.values, params, x=log_fid.qec_round.values, min_qec=3)

MAX_QEC = min(len(log_fid.log_fid), 60)

ax = out.plot_fit()
ax.plot(log_fid.qec_round.values[:MAX_QEC], log_fid.log_fid.values[:MAX_QEC], "b.", markersize=10)
ax.plot(MWPM_qec_round[:MAX_QEC], MPWM_log_fid[:MAX_QEC], "r.", markersize=10, label="MWPM")
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity")
ax.set_xticks(log_fid.qec_round.values[::2], log_fid.qec_round.values[::2])
ax.set_yticks(np.arange(0.5, 1, 0.05), np.round(np.arange(0.5, 1, 0.05), decimals=2))
ax.set_xlim(0, MAX_QEC+0.5)
ax.legend()
ax.grid(which='major')

print("logical error rate per cycle =", lmfit_par_to_ufloat(out.params["error_rate"]))

# %%
