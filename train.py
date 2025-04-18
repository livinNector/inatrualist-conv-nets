import argparse

from typing import Literal
import torch.nn as nn
from trainer import BaseModule, train

activation_funcs = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


class ConvNN(BaseModule):
    def __init__(
        self,
        input_shape: tuple[int, int, int],  # rgb image
        n_classes: int,
        n_conv_layers: int,
        max_n_filters: int,
        filter_organization: Literal["same", "double", "half"],
        kernel_size: int,
        activation_func: Literal["relu", "gelu", "silu", "mish", "leaky_relu", "elu"],
        conv_dropout=0.2,
        use_batch_norm=True,
        dense_size=512,
        dense_dropout=0.5,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_shape = input_shape
        self.output_shape = n_classes

        self.conv_blocks = nn.Sequential()
        in_channels = input_shape[0]

        if filter_organization == "same":
            layerwise_n_filters = [max_n_filters] * n_conv_layers
        elif filter_organization == "double":
            layerwise_n_filters = [
                max_n_filters // (2**i) for i in range(n_conv_layers)
            ]
            # reversing to introduce doubling
            layerwise_n_filters = layerwise_n_filters[::-1]
        elif filter_organization == "half":
            layerwise_n_filters = [
                max_n_filters // (2**i) for i in range(n_conv_layers)
            ]
        else:
            raise ValueError(
                'Filter organization should be one of "same", "double" or "half".'
            )

        activation = activation_funcs[activation_func]
        for i in range(n_conv_layers):
            n_filters = layerwise_n_filters[i]
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    n_filters,
                    kernel_size=kernel_size,
                    padding="same",
                ),
            )
            if use_batch_norm:
                block.append(nn.BatchNorm2d(n_filters))
            block.append(
                activation(inplace=True) if activation_func != "gelu" else activation()
            )
            block.append(nn.MaxPool2d(2))
            if conv_dropout > 0:
                block.append(nn.Dropout2d(conv_dropout))
            self.conv_blocks.append(block)
            in_channels = n_filters

        h, w = input_shape[1], input_shape[2]
        h_out, w_out = h // (2**n_conv_layers), w // (2**n_conv_layers)
        flattened_size = n_filters * h_out * w_out
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, dense_size),  # dense layer
            nn.BatchNorm1d(dense_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dense_dropout),
            nn.Linear(dense_size, self.output_shape),  # output layer
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x


def main(
    data_dir,
    max_n_filters: int = 256,
    filter_organization: Literal["same", "double", "half"] = "double",
    kernel_size: int = 3,
    activation_func: Literal[
        "relu", "gelu", "silu", "mish", "leaky_relu", "elu"
    ] = "relu",
    use_batch_norm=True,
    dense_size=512,
    dense_dropout=0.5,
    conv_dropout=0.3,
    augment=True,
):
    model = ConvNN(
        input_shape=(3, 244, 244),  # image net input size
        n_classes=10,
        n_conv_layers=5,
        max_n_filters=max_n_filters,
        filter_organization=filter_organization,
        kernel_size=kernel_size,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm,
        dense_size=dense_size,
        dense_dropout=dense_dropout,
        conv_dropout=conv_dropout,
    )

    run_name = "_".join(
        [
            f"{k}_{v}"
            for k, v in [
                ("nf", max_n_filters),
                ("fo", filter_organization),
                ("k", kernel_size),
                ("d", dense_size),
                ("bn", use_batch_norm),
                ("aug", augment),
                ("drop", conv_dropout),
            ]
        ]
    )
    train(model, data_dir, augment, n_epochs=10, run_name=run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_n_filters", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument(
        "--activation_func", type=str, choices=activation_funcs.keys(), default="relu"
    )
    parser.add_argument(
        "--filter_organization",
        type=str,
        choices=["same", "double", "half"],
        default="same",
    )
    parser.add_argument("--use_batch_norm", type=bool, default=True)
    parser.add_argument("--conv_dropout", type=float, default=0)
    parser.add_argument("--dense_size", type=int, default=512)
    parser.add_argument("--dense_dropout", type=int, default=0.4)
    parser.add_argument("--augment", type=bool, default=True)

    args = parser.parse_args()
    main(**dict(args._get_kwargs()))
