import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch import nn, cat, no_grad, cuda, float as tf_float, tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

MAX_SEQ_LENGTH = 300
HIDDEN_DIM = 512
MLP_DIM = 1024  # 2 layers
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NOT_METADATA_COLS = [
    "body",
    "title",
    "label",
    "light_cleaned_title",
    "light_cleaned_body",
    "cleaned_title",
    "cleaned_body",
]
DEVICE = "cuda" if cuda.is_available() else "cpu"


class TensorIndexDataset(TensorDataset):
    def __getitem__(self, index):
        """
        Returns in addition to the actual data item also its index (useful when assign a prediction to a item)
        """
        return index, super().__getitem__(index)


class BertMultiClassifier(nn.Module):
    def __init__(self, labels_count, hidden_dim=768, dropout=0.1):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, labels_count)
        self.softmax = nn.Softmax()

    def forward(self, tokens, masks):
        _, pooled_output = self.bert(
            tokens, attention_mask=masks, output_all_encoded_layers=False
        )
        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)
        proba = self.softmax(linear_output)

        return proba


class ExtraBertMultiClassifier(nn.Module):
    def __init__(
        self, labels_count, hidden_dim=768, mlp_dim=100, extras_dim=0, dropout=0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count),
        )
        self.softmax = nn.Softmax()

    def forward(self, tokens, masks, extras):
        _, pooled_output = self.bert(
            tokens, attention_mask=masks, output_all_encoded_layers=False
        )
        dropout_output = self.dropout(pooled_output)

        concat_output = cat((dropout_output, extras), dim=1)
        mlp_output = self.mlp(concat_output)
        proba = self.softmax(mlp_output)

        return proba


def train(model, optimizer, train_dataloader, metadata=None):

    for epoch_num in range(NUM_TRAIN_EPOCHS):
        print(f"Epoch {epoch_num + 1}")
        model.train()
        train_loss = 0

        for step_num, batch_data in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if metadata:
                token_ids, masks, extras, true_labels = tuple(
                    t.to(DEVICE) for t in batch_data
                )
                probas = model(token_ids, masks, extras)
            else:
                token_ids, masks, true_labels = tuple(t.to(DEVICE) for t in batch_data)
                probas = model(token_ids, masks)

            loss_func = nn.BCELoss()
            batch_loss = loss_func(probas[0], true_labels)
            train_loss += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch_num + 1} loss: {train_loss / (step_num + 1)}")

    return model


def score(model, data_loader, metadata=None):

    model.eval()
    output_ids = []
    outputs = None

    with no_grad():
        for step_num, batch_item in enumerate(data_loader):
            batch_ids, batch_data = batch_item

            if metadata:
                token_ids, masks, extras, _ = tuple(t.to(DEVICE) for t in batch_data)
                logits = model(token_ids, masks, extras)

            else:
                token_ids, masks, _ = tuple(t.to(DEVICE) for t in batch_data)
                logits = model(token_ids, masks)

            logits_np = logits.cpu().detach().numpy()

            if not outputs:
                outputs = logits_np
            else:
                outputs = np.vstack((outputs, logits_np))

            output_ids += batch_ids.tolist()

        print("Evaluation complete")

        return output_ids, outputs


def get_model(labels, metadata_cols=None):

    if metadata_cols:
        model = ExtraBertMultiClassifier(
            labels_count=len(labels),
            hidden_dim=HIDDEN_DIM,
            extras_dim=len(metadata_cols),
            mlp_dim=MLP_DIM,
        )
    else:
        model = BertMultiClassifier(labels_count=len(labels), hidden_dim=HIDDEN_DIM,)

    return model


def text_to_train_tensors(texts, tokenizer, max_seq_length):
    train_tokens = list(
        map(lambda t: ["[CLS]"] + tokenizer.tokenize(t)[: max_seq_length - 1], texts)
    )
    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    train_tokens_ids = pad_sequences(
        train_tokens_ids,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        dtype="int",
    )

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]

    return tensor(train_tokens_ids), tensor(train_masks)


def convert_df_to_tensor(df, dataset_cls, metadata_cols=None):
    texts = [
        t + ".\n" + df["light_cleaned_body"] for t in df["light_cleaned_title"].values
    ]
    y_vals = df["label"].values
    train_y_tensor = tensor(y_vals).float()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if metadata_cols:
        metadata = df[metadata_cols].values

        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(
            texts, tokenizer, MAX_SEQ_LENGTH
        )
        train_extras_tensor = tensor(metadata, dtype=tf_float)

        train_dataset = dataset_cls(
            train_tokens_tensor, train_masks_tensor, train_extras_tensor, train_y_tensor
        )

    else:
        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(
            texts, tokenizer, MAX_SEQ_LENGTH
        )
        train_dataset = dataset_cls(
            train_tokens_tensor, train_masks_tensor, train_y_tensor
        )

    return DataLoader(train_dataset, batch_size=BATCH_SIZE)


def run_model_fit():
    df = pd.read_parquet(PREPROCESSING_PATH)
    labels = df["label"].unique()
    metadata_cols = [x for x in df.colums if x not in NOT_METADATA_COLS]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    model = get_model(labels)

    if DEVICE == "cuda":
        model.cuda()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Train WITHOUT metadata
    train_dataloader = convert_df_to_tensor(train_df, TensorDataset)
    trained_model = train(model, optimizer=optimizer, train_dataloader=train_dataloader)

    score_dataloader = convert_df_to_tensor(val_df, TensorIndexDataset)
    output_ids, output = score(trained_model, score_dataloader)

    # Train WITH metadata
    train_dataloader = convert_df_to_tensor(train_df, TensorDataset, metadata_cols=metadata_cols)
    trained_model = train(model, optimizer=optimizer, train_dataloader=train_dataloader, metadata=metadata_cols)

    score_dataloader = convert_df_to_tensor(val_df, TensorIndexDataset, metadata_cols=metadata_cols)
    output_ids, output = score(trained_model, score_dataloader, metadata=metadata_cols)
