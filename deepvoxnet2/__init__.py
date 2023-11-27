import os

models_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "deepvoxnet2", "models")
DWI_model_dir = os.path.join(models_dir, 'DWI')
NCCT_model_dir = os.path.join(models_dir, 'NCCT')
FLAIR_model_dir = os.path.join(models_dir, 'FLAIR')
