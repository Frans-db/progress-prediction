from video_dataset import VideoFrameDataset
import matplotlib.pyplot as plt

def main():
    dataset = VideoFrameDataset(
            './data/MNIST/toy', 
            './data/MNIST/annotations.txt',
            num_segments=1,
            frames_per_segment=50,
            imagefile_template='img_{:05d}.png'
        )
    sample = dataset[0]
    frames = sample[0]
    labels = sample[1]
    print(labels)
    
if __name__ == '__main__':
    main()