import numpy as np
import matplotlib.pyplot as plt

def main():
    iterations = 100_000

    num_frames = 100
    num_segments = 1
    frames_per_segment = 5

    results = {}

    for _ in range(iterations):
        start_indices = get_indices(num_frames, num_segments, frames_per_segment)
        for index in start_indices:
            if index not in results:
                results[index] = 0
            results[index] += 1

    keys = list(range(num_frames))
    values = [results.get(key, 0) for key in keys]
    plt.suptitle('Distribution of Frame Sampling')
    plt.title('100 frames, 1 segment, 30 frames per segment')
    plt.xlabel('Frame Number')
    plt.ylabel('Times Sampled')
    plt.bar(keys, values)
    plt.show()

def get_indices(num_frames: int, num_segments: int, frames_per_segment: int, test_mode: bool = False) -> np.ndarray:
    """
    For each segment, choose a start index from where frames
    are to be loaded from.

    Args:
        num_frames: Number of frames the video has
    Returns:
        List of indices of where the frames of each
        segment are to be loaded from.
    """
    # choose start indices that are perfectly evenly spread across the video frames.
    if test_mode:
        distance_between_indices = (
            num_frames - frames_per_segment + 1) / float(num_segments)

        start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                    for x in range(num_segments)])
    # randomly sample start indices that are approximately evenly spread across the video frames.
    else:
        max_valid_start_index = (
            num_frames - frames_per_segment + 1) // num_segments

        start_indices = np.multiply(list(range(num_segments)), max_valid_start_index) + \
            np.random.randint(max_valid_start_index,
                                size=num_segments)
    
    indices = []
    for start_index in start_indices:
        for i in range(frames_per_segment):
            indices.append(start_index + i)
    return indices

if __name__ == '__main__':
    main()