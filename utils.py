import os
import pickle
import warnings

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


NUM_STATES = 35
MAX_LENGTH = 20

device = torch.device("mps")

warnings.filterwarnings(
    "ignore", message="Calling `map_elements` without specifying `return_dtype`"
)


class NextCharRNN(nn.Module):
    def __init__(
        self,
        input_size,
        category_size,
        output_size,
        hidden_size=64,
    ):
        super(NextCharRNN, self).__init__()
        self.to_hidden = nn.Linear(input_size + category_size, hidden_size)
        # self.layer = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.layer = nn.RNN(hidden_size, hidden_size, num_layers=1, batch_first=True)
        # self.layer = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        input_hidden = self.to_hidden(input)
        out, hidden = self.layer(input_hidden, hidden)
        output = self.fc(out)
        return output, hidden


def mask(name):
    mask_list = []
    for i in range(1, len(name) + 1):
        mask_list.append(name[:i])
    return mask_list


def load_vocab(df, vocab_file_name="vocab.pickle"):
    if os.path.exists(vocab_file_name):
        with open(vocab_file_name, "rb") as f:
            vocab_dict, rev_vocab_dict = pickle.load(f)
    else:
        vocab_set = set(df.select(pl.col("town_village_name").str.concat("")).item())
        vocab_dict = {"0": 0}
        rev_vocab_dict = {0: "0"}
        for i, c in enumerate(vocab_set):
            vocab_dict[c] = i + 1
            rev_vocab_dict[i + 1] = c
        with open(vocab_file_name, "wb") as f:
            pickle.dump((vocab_dict, rev_vocab_dict), f)

    return vocab_dict, rev_vocab_dict


def load_data():
    df = pl.read_excel("PC11_TV_DIR.xlsx")
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
    df = (
        df.with_columns(mask_list=pl.col("town_village_name").map_elements(mask))
        .explode(columns="mask_list")
        .with_columns(
            mask_list=pl.col("mask_list").str.zfill(MAX_LENGTH),
            mask_list_up=pl.col("mask_list")
            .shift(-1)
            .over(["town_village_name", "State Code"])
            .str.zfill(MAX_LENGTH),
        )
        .drop_nulls()
    )
    df = df.sample(fraction=1.0, shuffle=True, seed=42)
    return df


def map_vocab(x, vocab_dict):
    word_idx = [vocab_dict.get(i) for i in x]
    word_tensor = torch.tensor(word_idx, dtype=torch.long)
    word_one_hot = torch.nn.functional.one_hot(word_tensor, len(vocab_dict))
    return word_one_hot


def create_x(df, start, stop, vocab_dict):
    x = (
        df[start:stop]
        .select(pl.col("mask_list").map_elements(lambda x: map_vocab(x, vocab_dict)))
        .to_series()
        .to_list()
    )
    x = torch.stack(x).type(torch.float32)
    states = torch.tensor(
        df[start:stop].select((pl.col("State Code") - 1)).to_series().to_list(),
        dtype=torch.long,
    )
    states = torch.nn.functional.one_hot(states, num_classes=NUM_STATES)
    states = states.repeat(1, MAX_LENGTH).reshape(stop - start, MAX_LENGTH, NUM_STATES)
    x = torch.cat([x, states], dim=2)
    return x.to(device)


def split_data(df):
    train_df = df[: int(df.shape[0] * 0.8)]
    val_df = df[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.9)]
    test_df = df[int(df.shape[0] * 0.9) :]
    return train_df, val_df, test_df


def train_model(df, vocab_dict, rev_vocab_dict, batch_size, load=None):
    best_val_loss = 100000
    model = NextCharRNN(len(vocab_dict), NUM_STATES, len(vocab_dict))
    if load is not None:
        model.load_state_dict(torch.load(load))
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_df, val_df, test_df = split_data(df)
    print("START TRAIN")
    for epoch in range(500):
        print(f"{epoch=}")
        batch = 0
        train_loss = 0
        for start in range(0, train_df.shape[0], batch_size):
            batch += 1
            stop = min(start + batch_size, int(train_df.shape[0]))
            x = create_x(train_df, start, stop, vocab_dict)
            x = x.to(device)
            y = (
                train_df[start:stop]
                .select(
                    pl.col("mask_list_up").map_elements(
                        lambda x: map_vocab(x, vocab_dict)
                    )
                )
                .to_series()
                .to_list()
            )
            y = torch.stack(y).type(torch.float32).to(device)
            optimiser.zero_grad()
            output, hidden = model.forward(x)
            loss = criterion(output, y)
            train_loss += loss.item()
            writer.add_scalar("Loss/train", loss, epoch * train_df.shape[0] + start)
            loss.backward()
            optimiser.step()
            if batch % 50 == 0:
                print(f"{batch=}")
                generate_word(model, vocab_dict, rev_vocab_dict, inp="?", n=1)
                torch.save(model.state_dict(), f"models/model_{epoch}_{batch}.pt")
                print(loss)
        if epoch % 10 == 0:
            generate_word(model, vocab_dict, rev_vocab_dict, inp="?", n=1)
            torch.save(model.state_dict(), f"models/model_{epoch}_end.pt")
            print(loss)

        train_loss /= batch
        with torch.no_grad():
            val_loss = 0
            batch = 0
            for start in range(0, val_df.shape[0], batch_size):
                batch += 1
                stop = min(start + batch_size, val_df.shape[0])
                x = create_x(val_df, start, stop, vocab_dict)
                y = (
                    val_df[start:stop]
                    .select(
                        pl.col("mask_list_up").map_elements(
                            lambda x: map_vocab(x, vocab_dict)
                        )
                    )
                    .to_series()
                    .to_list()
                )
                y = torch.stack(y).type(torch.float32).to(device)

                output, _ = model.forward(x)
                loss = criterion(output, y)
                writer.add_scalar("Loss/val", loss, epoch * val_df.shape[0] + start)
                val_loss += loss.item()

        val_loss /= batch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pt")

        with torch.no_grad():
            batch = 0
            for start in range(0, test_df.shape[0], batch_size):
                batch += 1
                stop = min(start + batch_size, test_df.shape[0])
                x = create_x(test_df, start, stop, vocab_dict)
                y = (
                    test_df[start:stop]
                    .select(
                        pl.col("mask_list_up").map_elements(
                            lambda x: map_vocab(x, vocab_dict)
                        )
                    )
                    .to_series()
                    .to_list()
                )
                y = torch.stack(y).type(torch.float32).to(device)

                output, _ = model.forward(x)
                loss = criterion(output, y)
                writer.add_scalar("Loss/test", loss, epoch * test_df.shape[0] + start)

    return model


def generate_word(model, vocab_dict, rev_vocab_dict, inp="?", n=1):
    h = None
    for _ in range(n):
        cnt = 0
        random_state = torch.nn.functional.one_hot(
            torch.randint(0, NUM_STATES, (1,)), NUM_STATES
        )
        random_state = (
            random_state.repeat(1, MAX_LENGTH)
            .reshape(MAX_LENGTH, NUM_STATES)
            .to(device)
        )
        while True and cnt < MAX_LENGTH:
            padded_inp = (["0"] * (MAX_LENGTH - len(inp))) + list(inp)
            inp_tens = map_vocab(padded_inp, vocab_dict).to(device)
            inp_tens = torch.cat([inp_tens, random_state], dim=1)
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
