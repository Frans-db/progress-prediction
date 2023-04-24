import os

jobs = os.listdir('./')
jobs = sorted([f'sbatch {job}' for job in jobs if 'sbatch' in job])

print(' && '.join(jobs))
