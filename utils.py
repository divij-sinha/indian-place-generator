import os
import pickle
import warnings

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


state_code = {
    "Jammu & Kashmir": 1,
    "Himachal Pradesh": 2,
    "Punjab": 3,
    "Chandigarh": 4,
    "Uttarakhand": 5,
    "Haryana": 6,
    "Nct Of Delhi": 7,
    "Rajasthan": 8,
    "Uttar Pradesh": 9,
    "Bihar": 10,
    "Sikkim": 11,
    "Arunachal Pradesh": 12,
    "Nagaland": 13,
    "Manipur": 14,
    "Mizoram": 15,
    "Tripura": 16,
    "Meghalaya": 17,
    "Assam": 18,
    "West Bengal": 19,
    "Jharkhand": 20,
    "Odisha": 21,
    "Chhattisgarh": 22,
    "Madhya Pradesh": 23,
    "Gujarat": 24,
    "Daman & Diu": 25,
    "Dadra & Nagar Haveli": 26,
    "Maharashtra": 27,
    "Andhra Pradesh": 28,
    "Karnataka": 29,
    "Goa": 30,
    "Lakshadweep": 31,
    "Kerala": 32,
    "Tamil Nadu": 33,
    "Puducherry": 34,
    "Andaman & Nicobar Islands": 35,
}

state_code_rev = {
    1: "Jammu & Kashmir",
    2: "Himachal Pradesh",
    3: "Punjab",
    4: "Chandigarh",
    5: "Uttarakhand",
    6: "Haryana",
    7: "Nct Of Delhi",
    8: "Rajasthan",
    9: "Uttar Pradesh",
    10: "Bihar",
    11: "Sikkim",
    12: "Arunachal Pradesh",
    13: "Nagaland",
    14: "Manipur",
    15: "Mizoram",
    16: "Tripura",
    17: "Meghalaya",
    18: "Assam",
    19: "West Bengal",
    20: "Jharkhand",
    21: "Odisha",
    22: "Chhattisgarh",
    23: "Madhya Pradesh",
    24: "Gujarat",
    25: "Daman & Diu",
    26: "Dadra & Nagar Haveli",
    27: "Maharashtra",
    28: "Andhra Pradesh",
    29: "Karnataka",
    30: "Goa",
    31: "Lakshadweep",
    32: "Kerala",
    33: "Tamil Nadu",
    34: "Puducherry",
    35: "Andaman & Nicobar Islands",
}

MAX_LENGTH = 20
EPOCHS = 10
BATCH_SIZE = 100

if not torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

warnings.filterwarnings(
    "ignore", message="Calling `map_elements` without specifying `return_dtype`"
)


class NextCharRNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
    ):
        super(NextCharRNN, self).__init__()
        self.to_hidden = nn.Linear(input_size, hidden_size)
        # self.layer = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        # self.layer = nn.RNN(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.layer = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        input_hidden = self.to_hidden(input)
        out, hidden = self.layer(input_hidden, hidden)
        output = self.fc(out)
        return output, hidden


def load_vocab(df, vocab_file_name="static/vocab.pickle"):
    if os.path.exists(vocab_file_name):
        with open(vocab_file_name, "rb") as f:
            vocab_dict, rev_vocab_dict = pickle.load(f)
    else:
        vocab_set = set(df)
        vocab_dict = {}
        rev_vocab_dict = {}
        for i, c in enumerate(vocab_set):
            vocab_dict[c] = i
            rev_vocab_dict[i] = c
        with open(vocab_file_name, "wb") as f:
            pickle.dump((vocab_dict, rev_vocab_dict), f)

    return vocab_dict, rev_vocab_dict


def load_data():
    df = pl.read_excel("static/PC11_TV_DIR.xlsx")
    df = df.with_columns(
        pl.col("State Code").cast(pl.Int32),
        town_village_name="?"
        + pl.col("Town-Village Name")
        .str.to_lowercase()
        .str.replace_all("[\xa0ïü]", "", literal=False)
        .str.replace_all("[^a-z\\s]", "", literal=False)
        .str.replace_all("[\\s]+", " ", literal=False)
        .str.replace("  ", " ")
        + "!",
    )
    df = df.select("town_village_name", "State Code").unique()
    df_non_sel = df.filter(pl.col("town_village_name").str.len_chars() > MAX_LENGTH - 2)
    print(f"NOT SELECTING LENGTH >{MAX_LENGTH-2}, {df_non_sel.shape[0]} filtered")
    df = df.filter(pl.col("town_village_name").str.len_chars() <= MAX_LENGTH - 2)
    df = df.sample(fraction=1.0, shuffle=True, seed=42).select(
        (pl.col("State Code").cast(pl.String) + pl.col("town_village_name")).str.concat(
            "~"
        )
    )
    return df


def map_vocab(x, vocab_dict):
    word_idx = [vocab_dict.get(i) for i in x]
    word_tensor = torch.tensor(word_idx, dtype=torch.long)
    word_one_hot = torch.nn.functional.one_hot(word_tensor, len(vocab_dict))
    return word_one_hot.type(torch.float32).to(device)


def split_data(df):
    train_df = df[: int(len(df) * 0.8)]
    val_df = df[int(len(df) * 0.8) :]
    # test_df = df[int(len(df) * 0.9) :]
    return train_df, val_df, None


def train_model(df, vocab_dict, rev_vocab_dict, batch_size, load=None):
    best_val_loss = 100000
    model = NextCharRNN(len(vocab_dict), len(vocab_dict))
    if load is not None:
        model.load_state_dict(torch.load(load))
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_df, val_df, test_df = split_data(df)
    print("START TRAIN")
    for epoch in range(EPOCHS):
        print(f"{epoch=}")
        batch = 0
        train_loss = 0
        for start in range(0, len(train_df) - 1, batch_size):
            batch += 1
            stop = min(start + batch_size, int(len(train_df) - 1))
            x = map_vocab(train_df[start:stop], vocab_dict)
            y = map_vocab(train_df[start + 1 : stop + 1], vocab_dict)
            optimiser.zero_grad()
            output, hidden = model.forward(x)
            loss = criterion(output, y)
            train_loss += loss.item()
            writer.add_scalar("Loss/train", loss, epoch * len(train_df) + start)
            loss.backward()
            optimiser.step()
            if batch % 50 == 0:
                print(f"{batch=}")
                generate_word(model, vocab_dict, rev_vocab_dict, n=1)
                torch.save(model.state_dict(), f"models/model_{epoch}_{batch}.pt")
        if epoch % 10 == 0:
            generate_word(model, vocab_dict, rev_vocab_dict, n=1)
            torch.save(model.state_dict(), f"models/model_{epoch}_end.pt")

        train_loss /= batch
        with torch.no_grad():
            val_loss = 0
            batch = 0
            for start in range(0, int(len(val_df) - 1), batch_size):
                batch += 1
                stop = min(start + batch_size, int(len(val_df) - 1))
                x = map_vocab(val_df[start:stop], vocab_dict)
                y = map_vocab(val_df[start + 1 : stop + 1], vocab_dict)
                output, _ = model.forward(x)
                loss = criterion(output, y)
                writer.add_scalar("Loss/val", loss, epoch * len(val_df) + start)
                val_loss += loss.item()

        val_loss /= batch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pt")

    return model


def generate_word(model, vocab_dict, rev_vocab_dict, inp=None, n=1):
    h = None
    for _ in range(n):
        if inp is None:
            inp = "~" + str(np.random.randint(35)) + "?"
        print(inp, end="")
        cnt = 0
        while True and cnt < MAX_LENGTH:
            inp_tens = map_vocab(inp, vocab_dict).to(device)
            with torch.no_grad():
                cnt += 1
                o, h = model.forward(inp_tens.type(torch.float32))
                p = torch.softmax(o[-1], dim=0).cpu().numpy()
                idx = np.random.choice(len(rev_vocab_dict.keys()), p=p)
                tp = rev_vocab_dict[idx]
                if tp == "!":
                    break
                print(tp, end="")
                inp = inp + tp
        print("")


@torch.no_grad()
def generate_word_from_save(start_text, n, states, model_path=None, vocab_path=None):
    if vocab_path is None:
        vocab_path = "static/vocab.pickle"
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab_dict, rev_vocab_dict = pickle.load(f)
    if model_path is None:
        model_path = "models/best_model.pt"
    model = NextCharRNN(len(vocab_dict), len(vocab_dict))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    generations = []
    states = [s for s in states if s != 0]
    for _ in range(n):
        inp = "?" + start_text
        if len(states) > 0:
            state = np.random.choice(states)
            inp = "~" + str(state) + inp
        while True and len(inp) < MAX_LENGTH:
            inp_tens = map_vocab(inp, vocab_dict).to(torch.device("cpu"))
            o, h = model.forward(inp_tens.type(torch.float32))
            p = torch.softmax(o[-1], dim=0).cpu().numpy()
            idx = np.random.choice(len(rev_vocab_dict.keys()), p=p)
            tp = rev_vocab_dict[idx]
            if tp == "!":
                break
            inp = inp + tp

        generations.append(inp)
    return generations


if __name__ == "__main__":
    df = load_data()
    vocab, rev_vocab = load_vocab(df.item())
    model = train_model(df.item(), vocab, rev_vocab, batch_size=BATCH_SIZE)
