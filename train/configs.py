FBG_FOG_PARAMS = {
    "fbg": {
        "pose_length": 101,
        "skeleton_input_dim": 51,
        "skeleton_output_dim": 3,
        "sensor_in_channels": 3,
        "sensor_out_channels": 3,
        "sensor_length": 65,
        "shared_out_channels": 16,
        "backbone_dim": 8,
        "taskhead_input_dim": 8 * 16,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 256,
    },
    "fog": {
        "pose_length": 101,
        "skeleton_input_dim": 21,
        "skeleton_output_dim": 6,
        "sensor_in_channels": 6,
        "sensor_out_channels": 6,
        "sensor_length": 426,
        "shared_out_channels": 16,
        "backbone_dim": 8,
        "taskhead_input_dim": 8 * 16,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 256,
    },
}

FBG_FOG_PARAM_PRESETS = {
    "fog_sync_best_param": {
        "dataset": "fog",
        "sync_required": True,
        "description": "FoG sync setting: multimodal class-weighted loss, one shared task head, no train oversampling.",
        "train_params": {
            "epochs": 100,
            "batch_size": 256,
        },
        "args": {
            "wm": "class_wt",
            "alpha": 0.0,
            "consistency_lambda": 0.0,
            "use_norm_and_cos": False,
            "legacy_gcl_head": False,
            "sync_train_oversample": False,
        },
    },
    "fog_sync_trip_ce_alpha09": {
        "dataset": "fog",
        "sync_required": True,
        "description": "FoG sync no-class-rebalancing ablation: CE + CAGrad alpha=0.9, one shared task head, no train oversampling.",
        "train_params": {
            "epochs": 100,
            "batch_size": 256,
        },
        "args": {
            "wm": "ce",
            "alpha": 0.9,
            "consistency_lambda": 0.0,
            "use_norm_and_cos": False,
            "legacy_gcl_head": False,
            "sync_train_oversample": False,
        },
    },
    "fog_best_param": {
        "dataset": "fog",
        "description": "Final FoG async ours: old-ablation cosine-head training with tuned CAGrad/GCL.",
        "train_params": {
            "epochs": 50,
            "batch_size": 256,
        },
        "args": {
            "wm": "gcl",
            "alpha": 0.3,
            "max_norm": 0.0,
            "sgd_momentum": 0.9,
            "weight_decay": 1e-4,
            "patience": 100,
            "consistency_lambda": 1.0,
            "gcl_s": 40,
            "gcl_m": 0.4,
            "noise_mul": 0.0,
            "drw_warmup": 5,
            "use_norm_and_cos": True,
            "legacy_gcl_head": False,
        },
    },
}

WEARGAIT_GRAV = 9.81
WEARGAIT_ROOT = "data/WearGait"
WEARGAIT_HC_PATH = f"{WEARGAIT_ROOT}/HC"
WEARGAIT_PD_PATH = f"{WEARGAIT_ROOT}/PD"
WEARGAIT_HC_DEMO_CSV = f"{WEARGAIT_HC_PATH}/hc_demographic.csv"
WEARGAIT_PD_DEMO_CSV = f"{WEARGAIT_PD_PATH}/pd_demographic.csv"
WEARGAIT_PREPROC_DIR = f"{WEARGAIT_ROOT}/WearGait_preproc_SPmT_30Hz"
WEARGAIT_CSV_PATTERN = "*_SelfPace_matTURN.csv"

WEARGAIT_IMU_SITES = [
    "L_Ankle",
    "R_Ankle",
    "L_DorsalFoot",
    "R_DorsalFoot",
    "L_MidLatThigh",
    "R_MidLatThigh",
    "L_LatShank",
    "R_LatShank",
]

WEARGAIT_INSOLE_NUMERIC = [
    "LTotalForce_BW",
    "RTotalForce_BW",
    "SumForce_BW",
    "LCoP_X",
    "LCoP_Y",
    "RCoP_X",
    "RCoP_Y",
    "Linsole_Acc_X",
    "Linsole_Acc_Y",
    "Linsole_Acc_Z",
    "Rinsole_Acc_X",
    "Rinsole_Acc_Y",
    "Rinsole_Acc_Z",
]

WEARGAIT_WALKWAY_NUMERIC = ["L Foot Pressure_BW", "R Foot Pressure_BW"]
WEARGAIT_IMU_AXES = ("E", "N", "U")
WEARGAIT_IMU_FIXED = [
    f"{site}_FreeAcc_{axis}"
    for site in WEARGAIT_IMU_SITES
    for axis in WEARGAIT_IMU_AXES
]
WEARGAIT_INSOLE_FIXED = WEARGAIT_INSOLE_NUMERIC[:]
WEARGAIT_WALKWAY_FIXED = WEARGAIT_WALKWAY_NUMERIC[:]

DATASET_ALIASES = {
    "fbg": "fbg",
    "fog": "fog",
    "weargait": "weargait",
    # Backward-compatible names used by older caches/raw folders.
    "walk": "fbg",
    "turn": "fog",
}

RAW_READER_DATASET = {
    "fbg": "walk",
    "fog": "turn",
    "weargait": "weargait",
}


def normalize_dataset_name(dataset: str) -> str:
    try:
        return DATASET_ALIASES[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset}") from exc


def raw_reader_dataset_name(dataset: str) -> str:
    return RAW_READER_DATASET[normalize_dataset_name(dataset)]

MODEL_KEYS = (
    "skeleton_input_dim",
    "skeleton_output_dim",
    "sensor_in_channels",
    "sensor_out_channels",
    "sensor_length",
    "shared_out_channels",
    "backbone_dim",
    "taskhead_input_dim",
    "num_classes",
)
