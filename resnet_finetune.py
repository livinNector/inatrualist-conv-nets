import argparse
from typing import Literal

import torch
from collections import OrderedDict
from torch import nn
from torchvision import models

from trainer import BaseModule, train


class ResNetFinetune(BaseModule):
    def __init__(
        self,
        n_classes,
        finetune_type: Literal["full", "only_fc", "last_k_conv"],
        k=None,
    ):
        """Finetuning ResNet50 model with different strategies."""
        super().__init__()
        self.save_hyperparameters()

        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features

        self.feature_extractor = nn.Sequential(
            OrderedDict(list(backbone.named_children())[:-1])
        )

        # replacing last layer for inaturalist output classes
        self.classifier = nn.Linear(num_filters, n_classes)

        if finetune_type == "last_k_conv" and (k < 1 or k > 4):
            raise ValueError("k should be between 1 to 4 (inclusive).")

        if finetune_type == "full":
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        elif finetune_type == "only_fc":
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        elif finetune_type == "last_k_conv":
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # This enables the last k conv blocks
            for i in range(4, 4 - k, -1):
                for param in self.feature_extractor.get_submodule(
                    f"layer{i}"
                ).parameters():
                    param.requires_grad = True

        # This is enabled for all
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet on iNaturalist")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--finetune_type",
        type=str,
        required=True,
        choices=["full", "only_fc", "last_k_conv"],
        help="Finetuning strategy",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of last conv blocks to train (for last_k_conv)",
    )
    parser.add_argument(
        "--augment", type=bool, default=True, help="Use data augmentation"
    )
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")

    parser.add_argument(
        "--wandb_project", type=str, default=None, help="The wandb Project"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = ResNetFinetune(n_classes=10, finetune_type=args.finetune_type, k=args.k)

    train(
        model=model,
        data_dir=args.data_dir,
        augment=args.augment,
        n_epochs=args.n_epochs,
        run_name=args.finetune_type,
        wandb_project=args.wandb_project,
        test=True,
    )


if __name__ == "__main__":
    main()
