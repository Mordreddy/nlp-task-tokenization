import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

#One-Hot Encoding
abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
abc = abc + abc.lower() + "_"

def letter_index(lett):
    if lett in abc:
        return abc.find(lett)
    return abc.find("_")

def word_to_tensor(word):
    r = torch.zeros(len(word), len(abc))
    for index, lett in enumerate(word):
        r[index][letter_index(lett)] = 1
    return r

# PyTorch Dataset
class NameCountryDataset(Dataset):
    def __init__(self, data_dir="names"):
        self.COUNTRY_CLASSES = {
            "Arabic": 0, "Chinese": 1, "Czech": 2, "Dutch": 3,
            "English": 4, "French": 5, "German": 6, "Greek": 7,
            "Irish": 8, "Italian": 9, "Japanese": 10, "Korean": 11,
            "Polish": 12, "Portuguese": 13, "Russian": 14,
            "Scottish": 15, "Spanish": 16, "Vietnamese": 17
        }
        self.data = self._load_data(data_dir)
        self.num_countries = 18

    def _load_data(self, data_dir):
        dataset = []
        for file_path in glob.glob(os.path.join(data_dir, "*.txt")):
            country_name = os.path.splitext(os.path.basename(file_path))[0]
            country_idx = self.COUNTRY_CLASSES[country_name]
            with open(file_path, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            for name in names:
                dataset.append((name, country_idx))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, country_idx = self.data[idx]
        name_one_hot = word_to_tensor(name)
        country_vector = torch.zeros(self.num_countries)
        country_vector[country_idx] = 1.0
        return name_one_hot, country_vector


# Run
if __name__ == "__main__":
    dataset = NameCountryDataset(data_dir="names")

    # Test output
    test_input, test_output = dataset[0]
    print("Training Data：")
    print(f"Input (Name One-Hot): {test_input.shape}")
    print(f"Output (18D Country Vector): {test_output.shape}")
    print(f"Country Index: {torch.argmax(test_output).item()}")