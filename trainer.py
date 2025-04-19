import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms, datasets

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, F1Score, MetricCollection


def transform_augment(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(
                degrees=(-10, 10),
                translate=(0, 0.2),
                scale=(0.8, 1.2),
                shear=(-10, 10),
            ),
            transforms.ToTensor(),
        ]
    )


def transform_resize(image_size):
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )


class InaturalistDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir, val_size=0.2, augment=True, image_size=224, batch_size=32
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment
        self.image_size = image_size
        self.val_size = val_size

        self.train_transform = (
            transform_augment(image_size) if augment else transform_resize(image_size)
        )
        self.val_transform = transform_resize(image_size)

    def setup(self, stage=None):
        self.train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, "train"),
            transform=transform_augment(self.image_size)
            if self.augment
            else transform_resize(self.image_size),
        )
        self.test_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, "val"),
            transform=transform_resize(self.image_size),
        )
        targets = [label for _, label in self.train_dataset.samples]

        strat_split = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_size, random_state=1
        )
        train_idx, val_idx = next(strat_split.split([0] * len(targets), targets))

        self.train_subset = Subset(self.train_dataset, train_idx)
        self.val_subset = Subset(
            datasets.ImageFolder(
                os.path.join(self.data_dir, "train"),
                transform=transform_resize(self.image_size),
            ),
            val_idx,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy("multiclass", num_classes=10, average="macro"),
                "f1_score": F1Score("multiclass", num_classes=10, average="macro"),
            },
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {
                "accuracy": Accuracy("multiclass", num_classes=10, average="macro"),
                "f1_score": F1Score("multiclass", num_classes=10, average="macro"),
            },
            prefix="val_",
        )

        self.test_metrics = MetricCollection(
            {
                "accuracy": Accuracy("multiclass", num_classes=10, average="macro"),
                "f1_score": F1Score("multiclass", num_classes=10, average="macro"),
            },
            prefix="test_",
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)

        self.train_metrics.update(logits, y)

        self.log_dict(self.train_metrics.compute(), prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, logger=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.val_metrics.update(logits, y)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.test_metrics.update(logits, y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train(
    model, data_dir, augment, n_epochs, wandb_project=None, run_name=None, test=False
):
    torch.cuda.empty_cache()
    wandb_logger = WandbLogger(log_model="all", project=wandb_project)
    wandb_logger.experiment.name = run_name + "-" + wandb_logger.experiment.name
    data_module = InaturalistDataModule(
        data_dir=data_dir, val_size=0.2, augment=augment, image_size=224, batch_size=32
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy", mode="max", min_delta=0.01, patience=5
    )
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=20,
        val_check_interval=100,
        precision=16 if torch.cuda.is_available() else 32,
    )
    trainer.fit(model, datamodule=data_module)
    if test:
        trainer.test(model, datamodule=data_module)
