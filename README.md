# Is there progress in activity progress prediction?

Official PyTorch implementation of "Is there progress in activity progress prediction".

[[Arxiv](https://arxiv.org/abs/2308.05533)]

Please cite the paper when reporting, reproducing or extending the results.

```bibtex
@misc{deboer2023progress,
      title={Is there progress in activity progress prediction?}, 
      author={Frans de Boer and Jan C. van Gemert and Jouke Dijkstra and Silvia L. Pintea},
      year={2023},
      eprint={2308.05533},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Overview

This repository implements our paper "Is there progress in activity progress prediction". In it you can find implementations of 3 deep learning models from previous work: [ProgressNet](https://arxiv.org/abs/1705.01781), [RSDNet](https://arxiv.org/abs/1802.03243), and [UTE](https://arxiv.org/abs/1904.04189).

## Results

ProgressNet predictions on a video from the Golfswing activity for both full-video sequences (blue) and video-segments (orange) at timestamp t=125. The methods cannot learn when given video-segments.
![UCF101-24](./assets/results/ucf_video_GolfSwing_v_GolfSwing_g01_c02.png)

etwork predictions on a video from the ‘pancake’ cooking activity for all LSTM-based methods when given full-video sequences of random-noise at timestamp t=150. The networks cannot learn from the data and instead learn how to count.
![Breakfast](./assets/results/bf_video_pancake_P12_cam01.png)

Progress prediction example on Video-04 of Cholec80 at timestamp t=210. The methods recognize the surgical tool and correct their progress to signal the start of the procedure.
![Cholec80](./assets/results/cholec80_video04.png)

## Datasets

We performed our testing on [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), more specifically we used [these annotations for UCF101-24](https://github.com/gurkirt/corrected-UCF101-Annots) and downloaded the data from [here](https://github.com/gurkirt/realtime-action-detection), [Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/), and [Cholec80](http://camma.u-strasbg.fr/datasets). Furthermore, we test on a toy dataset called `Progress-bar` which can be generated with the code.

## Using the code

This code was made to be as versatile as possible, with the goal of analysing multiple deep learning models. This can make the code a bit difficult to read for use for an individual experiment, but most of the logic is contained in the following files:

`arguments.py`: This contains all the arguments that are used to setup an experiment. Each argument is explained in the file

`main.py`: This handles experiment setup.

`experiment.py`: Most of the main logic happens here. This contains the `Experiment` class. This class handles the training, testing, logging, and saving of an experiment.

`setup.py`: This creates randomized split files for Cholec80, and creates the backbone pth files for ResNet152, ResNet18, VGG16, and VGG11.

To run the code a dataset is needed. This dataset needs to be structured the same way as the UCF101-24 data which can be downloaded [here](https://github.com/gurkirt/realtime-action-detection).

Finally a syntethic dataset can be created using `make_bars.py`.

## Contact

If you have any questions feel free to contact me at [github.grumbly403@passmail.net](mailto:github.grumbly403@passmail.net) (private email generated using [Proton](https://proton.me/), any emails send to this are forwared to my real email address)
