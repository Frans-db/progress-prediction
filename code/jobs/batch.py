import os

root = './'
job_names = os.listdir('./')

for job_name in job_names:
    job_path = os.path.join(root, job_name)
    if not os.path.isdir(job_path):
        continue
    print(f'--- Job {job_name} ---')

    jobs = sorted(os.listdir(job_path))
    script = ' && '.join(jobs)
    print(script)
