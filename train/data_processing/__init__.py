_EXPORTS = {
    "MirrorReflection": (".common", "MirrorReflection"),
    "PoseSequenceAugmentation": (".common", "PoseSequenceAugmentation"),
    "RandomNoise": (".common", "RandomNoise"),
    "RandomRotation": (".common", "RandomRotation"),
    "axis_mask": (".common", "axis_mask"),
    "count_labels": (".common", "count_labels"),
    "extract_unique_subs": (".common", "extract_unique_subs"),
    "get_AMBID_from_Videoname": (".common", "get_AMBID_from_Videoname"),
    "load_reader": (".dataset_cache", "load_reader"),
    "reader_cache_path": (".dataset_cache", "reader_cache_path"),
    "visualize_sequence": (".common", "visualize_sequence"),
    "walkid_to_AMBID": (".common", "walkid_to_AMBID"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
