# A small VAE for Super Mario Bros levels

## Getting ready

Start by installing everything you need with

```
pip install -r requirements.txt
```

**I recommend creating your own Python environment first.** (something like `conda create -n "modernai" python=3.7`).

## Sampling from a trained model

I provide an already trained model in `./models/mario_vae_zdim_2_overfitted.pt`. Check `example_sampling_random_levels.py` to get a feel of how to load the model, query and visualize the levels in its latent space.

## Training a new model with a higher latent dimension

I also provide a script which allows for training a VAE on another latent dimension. An example on how to use it:

```
python train_vae.py --z-dim=32 --max-epochs=300 --overfit
```

I really recommend overfitting the network if you want to get interesting artifacts! This training process will save checkpoints of the VAE every 20 iterations. I would recommend using the one at iterations >100 (to make sure it's very overfitted).