import os

# Resolve paths relative to the repository, not the shell's current directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_ROOT = os.path.join(DATA_ROOT, 'raw')
LEGACY_PD_DATA_ROOT = os.path.join(PROJECT_ROOT, 'PD_3D_motion-capture_data')
PD_DATA_ROOT = os.path.join(RAW_DATA_ROOT, 'PD_3D_motion-capture_data')


def _resolve_pd_root() -> str:
	if os.path.exists(PD_DATA_ROOT):
		return PD_DATA_ROOT
	return LEGACY_PD_DATA_ROOT


def _resolve_data_path(*parts: str) -> str:
	for root in (_resolve_pd_root(), DATA_ROOT):
		path = os.path.join(root, *parts)
		if os.path.exists(path):
			return path
	return os.path.join(_resolve_pd_root(), *parts)


def _resolve_first_data_path(*candidates: tuple[str, ...]) -> str:
	for parts in candidates:
		path = _resolve_data_path(*parts)
		if os.path.exists(path):
			return path
	return _resolve_data_path(*candidates[0])


def get_pd_paths() -> dict:
	root = _resolve_pd_root()
	return {
		'walk': {
			'pose_path': _resolve_first_data_path(
				('FBG',),
				('C3Dfiles_processed_new',),
				('C3Dfiles_cleaned_sequences',),
			),
			'sensor_path': _resolve_data_path('GRF_processed'),
			'label_path': _resolve_data_path('PDGinfo.xlsx'),
		},
		'turn': {
			'pose_path': _resolve_first_data_path(
				('FoG', 'predictions'),
				('turn-in-place', 'predictions'),
			),
			'lifted_path': _resolve_first_data_path(
				('FoG', 'lifted'),
				('turn-in-place', 'lifted'),
			),
			'sensor_path': _resolve_first_data_path(
				('FoG', 'IMU'),
				('turn-in-place', 'IMU'),
			),
			'label_path': _resolve_first_data_path(
				('FoG', 'PDFEinfo.xlsx'),
				('turn-in-place', 'PDFEinfo.xlsx'),
			),
		},
	}


# Directory for preprocessed data.
PREPROCESSED_DATA_ROOT_PATH = os.path.join(_resolve_pd_root(), 'C3Dfiles_processed')

# PD
PD_PATH_POSES = get_pd_paths()['walk']['pose_path']
PD_PATH_SENSORS = get_pd_paths()['walk']['sensor_path']
PD_PATH_LABELS = get_pd_paths()['walk']['label_path']
