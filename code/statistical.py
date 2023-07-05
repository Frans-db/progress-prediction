import torch
import torch.nn as nn

def calc_baselines(train_lengths, test_lengths):
    max_length = max(train_lengths)
    loss = nn.L1Loss(reduction="sum")
    averages = torch.zeros(max(train_lengths))
    counts = torch.zeros(max(train_lengths))
    for length in train_lengths:
        progress = torch.arange(1, length + 1) / length
        averages[:length] += progress
        counts[:length] += 1
    averages = averages / counts

    index_loss, static_loss, random_loss, count = 0, 0, 0, 0
    for length in test_lengths:
        l = min(length, max_length)
        progress = torch.arange(1, length + 1) / length

        average_predictions = torch.ones(length)
        average_predictions[:l] = averages[:l]
        index_loss += loss(average_predictions * 100, progress * 100).item()
        static_loss += loss(torch.full_like(progress, 0.5) * 100, progress * 100).item()
        random_loss += loss(torch.rand_like(progress) * 100, progress * 100).item()

        count += length

    length = max(max(test_lengths), max_length)
    predictions = torch.ones(length)
    predictions[:max_length] = averages[:max_length]

    return predictions, index_loss / count, static_loss / count, random_loss / count

def main():
    train_lengths = [round(x) for x in torch.normal(100, 0, size=(1000,)).tolist()]
    test_lengths = [round(x) for x in torch.normal(100, 0, size=(100,)).tolist()]
    _, loss, _, _ = calc_baselines(train_lengths, test_lengths)
    print(loss)

    train_lengths = [round(x) for x in torch.normal(100, 10, size=(1000,)).tolist()]
    test_lengths = [round(x) for x in torch.normal(100, 10, size=(100,)).tolist()]
    _, loss, _, _ = calc_baselines(train_lengths, test_lengths)
    print(loss)

    train_lengths = [round(x) for x in torch.normal(1000, 250, size=(1000,)).tolist()]
    test_lengths = [round(x) for x in torch.normal(2000, 250, size=(100,)).tolist()]
    _, loss, _, _ = calc_baselines(train_lengths, test_lengths)
    print(loss)

if __name__ == '__main__':
    main()