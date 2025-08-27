from pytorch_lightning import Trainer, LightningDataModule
from torch.utils.data import DataLoader
from analysisgnn.data.datasets import *


class GraphDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=1, num_workers=4):
        super(GraphDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset.name

    def prepare_data(self):
        self.select_dataset()

    def setup(self, stage=None):
        dataset = self.select_dataset()
        idx = torch.randperm(len(dataset)).long()
        nidx = int(len(idx) * 0.7)
        self.train_idx = idx[:nidx]
        self.val_idx = idx[nidx:]
        self.dataset_train = dataset[self.train_idx]
        self.dataset_val = dataset[self.val_idx]
        self.dataset_predict = dataset[self.val_idx[:5]]

    def select_dataset(self):
        if self.dataset_name == "ASAPGraphPerformanceDataset":
            return ASAPGraphPerformanceDataset()
        elif self.dataset_name == "ASAPGraphAlignmentDataset":
            return ASAPGraphAlignmentDataset()
        elif self.dataset_name == "ASAPPitchSpellingDataset":
            return ASAPPitchSpellingDataset()
        elif self.dataset_name == "Bach370ChoralesGraphVoiceSeparationDataset":
            return Bach370ChoralesGraphVoiceSeparationDataset()
        elif self.dataset_name == "MCMAGraphVoiceSeparationDataset":
            return MCMAGraphVoiceSeparationDataset()
        elif self.dataset_name == "AugmentedNetChordGraphDataset":
            return AugmentedNetChordGraphDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=self.batch_size, num_workers=self.num_workers)


class RNNDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=10, num_workers=4):
        super(RNNDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        idx = torch.randperm(len(dataset)).long()
        nidx = int(len(idx)*0.7)
        self.train_idx = idx[:nidx]
        self.val_idx = idx[nidx:]

    def train_dataloader(self):
        return DataLoader(self.dataset[self.train_idx], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset[self.val_idx], batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset[torch.randperm(self.val_idx)[:2]], batch_size=self.batch_size, num_workers=self.num_workers)


class CNNDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=10, num_workers=4):
        super(CNNDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        idx = torch.randperm(len(dataset)).long()
        nidx = int(len(idx)*0.7)
        self.train_idx = idx[:nidx]
        self.val_idx = idx[nidx:]

    def train_dataloader(self):
        return DataLoader(self.dataset[self.train_idx], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset[self.val_idx], batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset[torch.randperm(self.val_idx)[:2]], batch_size=self.batch_size, num_workers=self.num_workers)



class StandardTrainer(Trainer):
    def __init__(self, epochs=10, batch_size=1, logger=None, num_workers=4, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        super(StandardTrainer, self).__init__(max_epochs=epochs, logger=logger, **kwargs)

    def fit(self, model, datamodule = None, train_dataloaders = None, val_dataloaders = None, ckpt_path = None):
        if hasattr(datamodule, "graphs"):
            datamodule = GraphDataModule(datamodule, batch_size=self.batch_size, num_workers=self.num_workers)
        if hasattr(datamodule, "scores"):
            datamodule = RNNDataModule(datamodule, batch_size=self.batch_size, num_workers=self.num_workers)
        if hasattr(datamodule, "pianorolls_dicts"):
            datamodule = CNNDataModule(datamodule, batch_size=self.batch_size, num_workers=self.num_workers)

        self._call_and_handle_interrupt(
            self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )

    def predict(self, model, datamodule = None, dataloaders = None, return_predictions = True, ckpt_path = None):
        if hasattr(datamodule, "graphs"):
            datamodule = GraphDataModule(datamodule, batch_size=self.batch_size)
        if hasattr(datamodule, "scores"):
            datamodule = RNNDataModule(datamodule, batch_size=self.batch_size)
        if hasattr(datamodule, "pianorolls_dicts"):
            datamodule = CNNDataModule(datamodule, batch_size=self.batch_size)

        return self._call_and_handle_interrupt(
            self._predict_impl, model, dataloaders, datamodule, return_predictions, ckpt_path
        )





