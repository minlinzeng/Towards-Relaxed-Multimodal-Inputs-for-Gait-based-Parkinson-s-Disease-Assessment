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
