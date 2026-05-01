# 多分类任务：五个中文字中，“你”在第几位，就属于第几类 0~4

from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


DATA_PATH = Path("/Users/yifeifu/hub-TroE/fuyifei/week03/character_data.txt")

chars = list("山水花月风云海天星光夜色春秋青绿川河流明清高长远近")

def generate_balanced_file(path, n_per_class=200):
    lines = []

    for pos in range(5):
        for _ in range(n_per_class):
            seq = random.sample(chars, 4)
            seq.insert(pos, "你")
            lines.append("".join(seq))

    random.shuffle(lines)

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

class CharDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]

        self.samples = []
        all_chars = set()

        for line in lines:
            if len(line) != 5:
                continue
            if line.count("你") != 1:
                continue

            label = line.index("你")  # 0,1,2,3,4
            self.samples.append((line, label))
            all_chars.update(line)

        if not self.samples:
            raise ValueError("No valid samples found. Each line should have 5 chars and exactly one '你'.")

        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        for ch in sorted(all_chars):
            self.char2idx[ch] = len(self.char2idx)

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        return torch.tensor(
            [self.char2idx.get(ch, self.char2idx["<UNK>"]) for ch in text],
            dtype=torch.long
        )

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        x = self.encode(text)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)          # [batch, seq_len, embed_dim]
        output, hidden = self.rnn(x)   # hidden: [1, batch, hidden_dim]
        hidden = hidden[-1]            # [batch, hidden_dim]
        logits = self.fc(hidden)       # [batch, 5]
        return logits


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        hidden = hidden[-1]
        logits = self.fc(hidden)
        return logits


def train_model(model, train_loader, test_loader, num_epochs=30, lr=0.001):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, test_loader, device)
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Loss: {total_loss / len(train_loader):.4f}, "
                f"Test Acc: {acc:.4f}"
            )

    return model


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def predict(model, dataset, text):
    if len(text) != 5:
        raise ValueError("Input text must have exactly 5 Chinese characters.")

    device = next(model.parameters()).device
    model.eval()

    x = dataset.encode(text).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return pred


def main():
    random.seed(42)
    torch.manual_seed(42)

    generate_balanced_file(DATA_PATH, n_per_class=200)
    dataset = CharDataset(DATA_PATH)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    vocab_size = len(dataset.char2idx)

    # Choose one:
    # model = RNNClassifier(vocab_size)
    model = LSTMClassifier(vocab_size)

    model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=30,
        lr=0.001
    )

    examples = [
        "海青流秋你",
        "近你海清星",
        "花开你枝头",
        "你清光川明",
        "青山绿你水",
    ]

    print("\nInference:")
    for text in examples:
        pred = predict(model, dataset, text)
        print(f"{text} -> class {pred}, meaning position {pred + 1}")


if __name__ == "__main__":
    main()
