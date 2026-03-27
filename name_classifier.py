# Import required libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob

#Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
abc = abc + abc.lower() + "_"
N_LETTERS = len(abc)
MAX_NAME_LENGTH = 20

def letter_index(lett):
    if lett in abc:
        return abc.find(lett)
    return abc.find("_")

def word_to_tensor(word):
    tensor = torch.zeros(MAX_NAME_LENGTH, N_LETTERS)
    for li, letter in enumerate(word[:MAX_NAME_LENGTH]):
        tensor[li][letter_index(letter)] = 1
    return tensor

#2.PyTorch Dataset
class NamesDataset(Dataset):
    def __init__(self, data_dir="names"):
        # 18 Countries
        self.countries = [
            'Arabic', 'Chinese', 'Czech', 'Dutch', 'English',
            'French', 'German', 'Greek', 'Irish', 'Italian',
            'Japanese', 'Korean', 'Polish', 'Portuguese',
            'Russian', 'Scottish', 'Spanish', 'Vietnamese'
        ]
        self.country_to_idx = {c: i for i, c in enumerate(self.countries)}
        self.samples = self._load_data(data_dir)
        self.vocab_size = N_LETTERS
        self.max_len = MAX_NAME_LENGTH

    def _load_data(self, data_dir):
        samples = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, data_dir)

        # Load all txt files
        for file_path in glob.glob(os.path.join(full_path, "*.txt")):
            country = os.path.splitext(os.path.basename(file_path))[0]
            if country not in self.country_to_idx:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    # Clean name
                    cleaned = [c for c in name if c.isalpha()]
                    name = ''.join(cleaned).strip()
                    if name:
                        samples.append((name, self.country_to_idx[country]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, country_idx = self.samples[idx]
        name_tensor = word_to_tensor(name)
        return name_tensor, torch.tensor(country_idx, dtype=torch.long)

#3HIGH-PERFORMANCE MODEL
class NameClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Bidirectional 2-layer RNN with Dropout
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, hidden = self.rnn(x)
        # Concatenate forward and backward hidden states
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(out)
        return out

#4.TRAINING FUNCTION
def train_model(model, train_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = torch.argmax(y_pred, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

        # Print training stats
        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

#5. PREDICTION FUNCTION
def predict_name(model, name, dataset):
    model.eval()
    with torch.no_grad():
        # Convert input name to tensor
        x = word_to_tensor(name).unsqueeze(0).to(device)
        output = model(x)
        pred_idx = torch.argmax(output, dim=1).item()
    return dataset.countries[pred_idx]

if __name__ == '__main__':
    # 1. Load dataset
    dataset = NamesDataset(data_dir="names")
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} names | 18 countries")

    # 2. Initialize high-accuracy model
    model = NameClassifier(
        input_size=N_LETTERS,
        hidden_size=256,
        num_classes=len(dataset.countries)
    ).to(device)

    # 3. Train model
    print("\nStart training (20 epochs)")
    train_model(model, train_loader, epochs=20)

    # 4. Test prediction
    print("\nPrediction Results:")
    test_names = ["Cui", "Smith", "Mohammed", "Ivanov", "Nguyen"]
    for name in test_names:
        result = predict_name(model, name, dataset)
        print(f"Name: {name:10} → Country: {result}")