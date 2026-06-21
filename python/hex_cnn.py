"""Custom feature extractor for the spatial HexWars observation.

The env hands SB3 a FLAT float vector = [board planes (C*H*W), channel-major] + [G globals]. This
extractor reshapes the board part into a (C, H, W) image, runs a small CNN over it (so the policy can
exploit hex adjacency / spatial patterns), and concatenates the scalar globals before the
policy/value heads. Build the policy_kwargs from the server handshake via cnn_policy_kwargs().

Models trained with this are saved with the class reference, so any loader (duel/winrate/policy_server)
just needs `hex_cnn` importable — no extra wiring.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HexCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, channels, board_h, board_w, n_globals, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.c, self.h, self.w, self.g = channels, board_h, board_w, n_globals
        self.board_size = channels * board_h * board_w
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * board_h * board_w
        self.head = nn.Sequential(nn.Linear(conv_out + n_globals, features_dim), nn.ReLU())

    def forward(self, obs):
        board = obs[:, :self.board_size].reshape(-1, self.c, self.h, self.w)
        glob = obs[:, self.board_size:]
        return self.head(torch.cat([self.cnn(board), glob], dim=1))


def cnn_policy_kwargs(spaces_info):
    """Build MaskablePPO policy_kwargs from the server handshake (channels/board_h/board_w/globals)."""
    return dict(
        features_extractor_class=HexCNN,
        features_extractor_kwargs=dict(
            channels=int(spaces_info["channels"]),
            board_h=int(spaces_info["board_h"]),
            board_w=int(spaces_info["board_w"]),
            n_globals=int(spaces_info["globals"]),
        ),
    )
