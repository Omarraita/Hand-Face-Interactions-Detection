import torch
import torch.autograd as autograd
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence as PACK

from multiprocessing import cpu_count

import pytorch_lightning as pl

import math, copy
import sklearn
import numpy as np

def accuracy(pred, labels):
  return torch.sum(pred == labels)/len(labels)
  
class PadSequence:
  def __call__(self, batch):
    seqs = []
    labels = []
    lengths = []
    for tup in batch:
      seqs.append(tup["sequence"])
      labels.append(tup["label"])
      lengths.append(len(tup["sequence"]))
    
    indexes = np.argsort(-1*np.array(lengths))
    seqs = [seqs[i] for i in indexes]
    labels = np.array(labels)[indexes]
    lengths = np.array(lengths)[indexes]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    lengths = torch.Tensor([len(x) for x in seqs])
    #print('DIM 0',sorted_batch[0])
    #print('DIM 1',sorted_batch[1])
    labels = torch.LongTensor(labels)
    return sequences_padded, lengths, labels

class InteractionsDataset(Dataset):
  def __init__(self, sequences):
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequences, label = self.sequences[idx]
    seq = torch.Tensor(sequences.to_numpy())
    #return dict(sequence = F.pad(seq,(0,0,128-seq.size()[0],0)),
    return dict(sequence = seq,
    label = torch.tensor(label).long())
    #return seq, torch.tensor(label).long()


class InteractionsDataModule(pl.LightningDataModule):
  def __init__(self, train_sequences, test_sequences, batch_size):
    super().__init__()
    self.train_sequences = train_sequences
    self.test_sequences = test_sequences
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train_dataset = InteractionsDataset(self.train_sequences)
    self.test_dataset = InteractionsDataset(self.test_sequences)

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = cpu_count(),
        collate_fn=PadSequence()
    )
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = cpu_count(),
        collate_fn=PadSequence()
    )
  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = cpu_count(),
        collate_fn=PadSequence()
    )

class SequenceModel(nn.Module):

  def __init__(self, n_features, n_classes, n_hidden=256, n_layers=2):
    super().__init__()

    self.lstm = nn.LSTM(
        input_size = n_features,
        hidden_size=n_hidden,
        num_layers=n_layers,
        batch_first=True,
        dropout= 0.75
    )
    self.classifier = nn.Linear(n_hidden, n_classes)
  
  def forward(self, x):
    self.lstm.flatten_parameters()
    _, (hidden,_) = self.lstm(x)

    out = hidden[-1]
    return self.classifier(out)


class InteractionsPredictor(pl.LightningModule):
  def __init__(self, n_features : int, n_classes : int, train_len : int, test_len : int, bsz : int,
               lr = 0.0082):
    super().__init__()
    self.model = SequenceModel(n_features, n_classes)
    self.criterion = nn.CrossEntropyLoss()
    self.best_acc = 0
    self.best_f1 = 0
    self.best_weights = SequenceModel(n_features, n_classes)

    self.lr = lr
    self.bsz = bsz

    self.epoch_loss_train = []
    self.epoch_loss_val = []
    self.epoch_acc_train = []
    self.epoch_acc = []
    self.epoch_f1 = []

    self.loss_train = []
    self.loss_val = []
    self.acc_train = []
    self.acc_val = []
    self.f1 = []

    self.train_len = train_len
    self.test_len = test_len


  def forward(self, x, labels=None):
    output = self.model(x)
    loss = 0
    if labels is not None:
      loss = self.criterion(output, labels)
    return loss, output
  
  def training_step(self, batch, batch_idx):
    sequences, seq_lengths, labels = batch
    seq_pack = PACK(sequences, seq_lengths.to('cpu'), batch_first=True)
    loss, outputs = self(seq_pack, labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = sklearn.metrics.accuracy_score(labels.to('cpu').data.numpy().astype(int), predictions.to('cpu').data.numpy().astype(int))

    self.epoch_acc_train.append(step_accuracy)
    self.epoch_loss_train.append(loss.item())

    if batch_idx == math.floor((self.train_len)/self.bsz)-1:
      self.acc_train.append(np.mean(self.epoch_acc_train))
      self.loss_train.append(np.mean(self.epoch_loss_train))
      self.epoch_acc_train = []
      self.epoch_loss_train = []

    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def validation_step(self, batch, batch_idx):
    sequences, seq_lengths, labels = batch
    seq_pack = PACK(sequences, seq_lengths.to('cpu'), batch_first=True)
    loss, outputs = self(seq_pack, labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = sklearn.metrics.accuracy_score(labels.to('cpu').data.numpy().astype(int), predictions.to('cpu').data.numpy().astype(int))
    f1 = sklearn.metrics.f1_score(labels.to('cpu').data.numpy().astype(int), predictions.to('cpu').data.numpy().astype(int), average='macro')
    
    # Keep track of best model
    self.epoch_acc.append(step_accuracy)
    self.epoch_f1.append(f1)
    self.epoch_loss_val.append(loss.item())

    if batch_idx == math.floor((self.test_len)/self.bsz)-1:
      acc = np.mean(self.epoch_acc)
      f1 = np.mean(self.epoch_f1)
      loss_val = np.mean(self.epoch_loss_val)
      if acc >= self.best_acc:
        self.best_acc = acc
      if f1 >= self.best_f1:
        self.best_f1 = f1
      self.best_weights.load_state_dict(copy.deepcopy(self.model.state_dict()))

      # Save performance hist
      self.acc_val.append(acc)
      self.f1.append(f1)
      self.loss_val.append(loss_val)
      self.epoch_acc = []
      self.epoch_f1 = []
      self.epoch_loss_val = []
      
    # Log performance
    self.log("val_loss", loss, prog_bar=True, logger=True)
    self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
    self.log("F1 score", f1, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy, "f1": f1}

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=0.00082)