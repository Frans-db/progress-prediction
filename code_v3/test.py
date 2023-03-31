import wandb
import random

epochs = 10
wandb.init(
    project='wand-test',
    config={
        'learning_rate': 1e-3,
        'forecasting': False,
        'dataset': 'toy',
        'epochs': epochs,
    }
)

offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    wandb.log({
        'acc': acc,
        'loss': loss
    })

wandb.finish()

