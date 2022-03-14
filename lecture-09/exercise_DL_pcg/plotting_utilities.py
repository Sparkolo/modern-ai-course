from pathlib import Path

import PIL
import torch
import numpy as np

from vae_mario import VAEMario

filepath = Path(__file__).parent.resolve()
Tensor = torch.Tensor


def absolute(path_str):
    return str(Path(path_str).absolute())


encoding = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
}

sprites = {
    encoding["X"]: absolute(f"{filepath}/sprites/stone.png"),
    encoding["S"]: absolute(f"{filepath}/sprites/breakable_stone.png"),
    encoding["?"]: absolute(f"{filepath}/sprites/question.png"),
    encoding["Q"]: absolute(f"{filepath}/sprites/depleted_question.png"),
    encoding["E"]: absolute(f"{filepath}/sprites/goomba.png"),
    encoding["<"]: absolute(f"{filepath}/sprites/left_pipe_head.png"),
    encoding[">"]: absolute(f"{filepath}/sprites/right_pipe_head.png"),
    encoding["["]: absolute(f"{filepath}/sprites/left_pipe.png"),
    encoding["]"]: absolute(f"{filepath}/sprites/right_pipe.png"),
    encoding["o"]: absolute(f"{filepath}/sprites/coin.png"),
}


def get_img_from_level(level: np.ndarray):
    image = []
    for row in level:
        image_row = []
        for c in row:
            if c == encoding["-"]:  # There must be a smarter way than hardcoding this.
                # white background
                tile = (255 * np.ones((16, 16, 3))).astype(int)
            elif c == -1:
                # masked
                tile = (128 * np.ones((16, 16, 3))).astype(int)
            else:
                tile = np.asarray(PIL.Image.open(sprites[c]).convert("RGB")).astype(int)
            image_row.append(tile)
        image.append(image_row)

    image = [np.hstack([tile for tile in row]) for row in image]
    image = np.vstack([np.asarray(row) for row in image])

    return image


def plot_decoded_level(level: Tensor) -> np.ndarray:
    """
    If {level} is in one-hot encoding, returns
    the RGB that represents the level.
    """
    level_ = torch.argmax(level, dim=0).detach().numpy()
    return get_img_from_level(level_)


def plot_level_from_z(z: Tensor, vae: VAEMario) -> np.ndarray:
    """
    Returns the level in RBG
    """
    level = vae.decode(z)
    return plot_decoded_level(level)
