#!/usr/bin/env python3
# train_sentiment.py
# Requisitos: torch, torchtext, scikit-learn, pandas

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.texts = df['review_text'].tolist()
        self.labels = df['label'].tolist()
        # Construir vocabulario
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for text in self.texts:
            for token in text.split():
                token = token.lower()
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1
        self.unk_idx = self.vocab['<UNK>']

    def encode(self, text):
        return [self.vocab.get(token.lower(), self.unk_idx) for token in text.split()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encode(self.texts[idx]), dtype=torch.long), self.labels[idx]

def pad_collate(batch):
    texts, labels = zip(*batch)
    lengths = [len(t) for t in texts]
    max_len = max(lengths)
    padded = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = self.fc(hidden[-1])
        return out

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for texts, labels, lengths in loader:
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',    required=True)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--epochs',       type=int,   default=5)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--model_out',    required=True)
    parser.add_argument('--log_out',      required=True)
    parser.add_argument('--metrics_out',  required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SentimentDataset(args.data_path)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

    model     = SentimentModel(len(dataset.vocab), 100, 128, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Entrenamiento
    with open(args.log_out, 'w') as log_file:
        for epoch in range(1, args.epochs+1):
            loss, acc = train(model, loader, optimizer, criterion, device)
            log = f'Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}'
            print(log)
            log_file.write(log + '\n')

    # Guardar modelo
    torch.save(model.state_dict(), args.model_out)

    # Evaluación
    report, cm = evaluate(model, loader, criterion, device)
    pd.DataFrame(report).transpose().to_csv(args.metrics_out)
    print("Entrenamiento y evaluación completados.")

if __name__ == '__main__':
    main()