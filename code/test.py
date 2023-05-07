names = ['rsd', 'i3d', 'indices']
progress = [
    [0.020, 0.020, 0.013, 0.024],
    [0.063, 0.138, 0.168, 0.077],
    [0.015, 0.016, 0.021, 0.018]
]
rsd = [
    [9.48, 7.03, 8.19, 9.87],
    [18.92, 13.87, 19.31, 19.03],
    [6.75, 5.89, 8.84, 7.92]
]

for (name, progress_values, rsd_values) in zip(names, progress, rsd):
    print(name, sum(progress_values) / 4, sum(rsd_values) / 4)