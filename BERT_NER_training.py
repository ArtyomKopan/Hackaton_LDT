import pandas as pd
import numpy as np
import re
import ast
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import joblib
from tqdm import tqdm
import gc
import os

warnings.filterwarnings("ignore")

class Config:
    model_name = "cointegrated/rubert-tiny2"
    max_length = 256
    hidden_dropout = 0.1

    batch_size = 32
    learning_rate = 2e-5
    epochs = 5
    patience = 3
    warmup_steps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

def load_data(path, sep=","):
    df = pd.read_csv(path, sep=sep)
    df["annotation"] = df["annotation"].apply(ast.literal_eval)
    return df

def tokenize_with_spans(text):
    if isinstance(text, np.str_):
        text = str(text)
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens

def assign_labels_to_tokens(text, spans):
    if isinstance(text, np.str_):
        text = str(text)

    tokens_with_pos = tokenize_with_spans(text)
    labels = ["O"] * len(tokens_with_pos)

    char_to_label = {}
    for span in spans:
        start = int(span[0]) if isinstance(span[0], (np.integer, np.int64)) else span[0]
        end = int(span[1]) if isinstance(span[1], (np.integer, np.int64)) else span[1]
        label = str(span[2]) if isinstance(span[2], np.str_) else span[2]

        for i in range(start, end):
            char_to_label[i] = label

    for i, (token, start, end) in enumerate(tokens_with_pos):
        label = char_to_label.get(start, "O")
        labels[i] = label

    tokens = [t[0] for t in tokens_with_pos]
    return tokens, labels

def safe_label_transform(labels, label_encoder):
    transformed = []
    for label in labels:
        label_str = str(label).strip()
        if label_str not in label_encoder.classes_ or label_str.isdigit():
            transformed.append("O")
        else:
            transformed.append(label_str)
    return transformed

class BERTNERDataset(Dataset):
    def __init__(self, df, tokenizer, label_encoder, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

        self.samples = []

        for _, row in tqdm(df.iterrows(), desc="Подготовка данных", total=len(df)):
            text = row["sample"]
            spans = row["annotation"]

            if isinstance(text, np.str_):
                text = str(text)

            try:
                tokens, labels = assign_labels_to_tokens(text, spans)
            except Exception as e:
                continue

            if len(tokens) != len(labels):
                continue

            safe_labels = safe_label_transform(labels, label_encoder)

            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors=None
            )

            word_ids = encoding.word_ids()
            aligned_labels = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_labels.append(self.label_encoder.transform([safe_labels[word_idx]])[0])
                else:
                    aligned_labels.append(-100)
                previous_word_idx = word_idx

            padding_length = self.max_length - len(encoding["input_ids"])
            if padding_length > 0:
                encoding["input_ids"] = encoding["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
                encoding["attention_mask"] = encoding["attention_mask"] + [0] * padding_length
                encoding["token_type_ids"] = encoding["token_type_ids"] + [0] * padding_length
                aligned_labels = aligned_labels + [-100] * padding_length
            else:
                encoding["input_ids"] = encoding["input_ids"][:self.max_length]
                encoding["attention_mask"] = encoding["attention_mask"][:self.max_length]
                encoding["token_type_ids"] = encoding["token_type_ids"][:self.max_length]
                aligned_labels = aligned_labels[:self.max_length]

            self.samples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "token_type_ids": encoding["token_type_ids"],
                "labels": aligned_labels,
                "original_text": text,
                "original_tokens": tokens
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": torch.LongTensor(sample["input_ids"]),
            "attention_mask": torch.LongTensor(sample["attention_mask"]),
            "token_type_ids": torch.LongTensor(sample["token_type_ids"]),
            "labels": torch.LongTensor(sample["labels"])
        }

class BERTForNER(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dropout=0.1):
        super(BERTForNER, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask, token_type_ids)

        # Вычисляем loss только для не-игнорируемых токенов
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def evaluate(model, dataloader, device, label_encoder):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []
    num_batches = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            num_batches += 1

            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100

            for i in range(labels.size(0)):
                seq_predictions = predictions[i][mask[i]]
                seq_labels = labels[i][mask[i]]

                all_predictions.extend(seq_predictions.cpu().numpy())
                all_true_labels.extend(seq_labels.cpu().numpy())

    if len(all_predictions) == 0:
        return 0, 0, 0, 0, 0

    try:
        all_predictions_decoded = label_encoder.inverse_transform(all_predictions)
        all_true_labels_decoded = label_encoder.inverse_transform(all_true_labels)
    except ValueError as e:
        print(f"Ошибка при декодировании: {e}")
        all_predictions_decoded = []
        all_true_labels_decoded = []

        for pred_idx in all_predictions:
            if pred_idx < len(label_encoder.classes_):
                all_predictions_decoded.append(label_encoder.classes_[pred_idx])
            else:
                all_predictions_decoded.append("O")

        for true_idx in all_true_labels:
            if true_idx < len(label_encoder.classes_):
                all_true_labels_decoded.append(label_encoder.classes_[true_idx])
            else:
                all_true_labels_decoded.append("O")

    accuracy = accuracy_score(all_true_labels_decoded, all_predictions_decoded)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels_decoded, all_predictions_decoded, average="macro", zero_division=0
    )

    return total_loss / num_batches, accuracy, precision, recall, f1

def predict_on_test_set(test_df, model, tokenizer, label_encoder, max_length, device):
    results = []

    model.eval()
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), desc="Предсказание", total=len(test_df)):
            text = row["sample"]
            if isinstance(text, np.str_):
                text = str(text)

            tokens_with_pos = tokenize_with_spans(text)
            tokens = [t[0] for t in tokens_with_pos]

            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            word_ids = encoding.word_ids()

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            token_type_ids = encoding["token_type_ids"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            predictions = torch.argmax(logits, dim=-1)

            word_predictions = []
            previous_word_idx = None

            for i, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx != previous_word_idx:
                    pred_idx = predictions[0, i].item()
                    if pred_idx < len(label_encoder.classes_):
                        word_predictions.append(label_encoder.classes_[pred_idx])
                    else:
                        word_predictions.append("O")
                    previous_word_idx = word_idx

            word_predictions = word_predictions[:len(tokens)]

            annotations = []
            for (token, start, end), label in zip(tokens_with_pos, word_predictions):
                if label != "O":
                    start_int = int(start) if isinstance(start, (np.integer, np.int64)) else start
                    end_int = int(end) if isinstance(end, (np.integer, np.int64)) else end
                    label_str = str(label) if isinstance(label, np.str_) else label
                    annotations.append((start_int, end_int, label_str))

            results.append({
                "sample": text,
                "annotation": annotations
            })

    return pd.DataFrame(results)

def analyze_labels(df, label_encoder, dataset_name):
    print(f"\nАнализ меток в {dataset_name}:")
    all_labels = []

    for _, row in df.iterrows():
        text = row["sample"]
        spans = row["annotation"]
        try:
            tokens, labels = assign_labels_to_tokens(text, spans)
            all_labels.extend([str(label).strip() for label in labels])
        except:
            continue

    unique_labels = set(all_labels)
    print(f"Уникальные метки: {unique_labels}")
    print(f"Метки, отсутствующие в encoder: {unique_labels - set(label_encoder.classes_)}")

    label_dist = {}
    for label in all_labels:
        label_dist[label] = label_dist.get(label, 0) + 1

    print(f"Распределение меток: {dict(sorted(label_dist.items(), key=lambda x: x[1], reverse=True))}")

def main():
    print(f"Используется устройство: {config.device}")

    print("\n1. Загрузка данных...")
    train_df = load_data("train_data_corrected.csv")
    dev_df = load_data("dev_data_corrected.csv")
    synthetic_train_df = load_data("new_synthetic_train_data_corrected.csv")
    synthetic_dev_df = load_data("new_synthetic_dev_data_corrected.csv")

    print("\n2. Загрузка токенизатора и модели...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    print(f"Токенизатор загружен: {config.model_name}")

    print("\n3. Подготовка меток...")
    all_labels = []
    label_counts = {}

    for _, row in tqdm(synthetic_train_df.iterrows(), desc="Сбор меток", total=len(train_df)):
        text = row["sample"]
        spans = row["annotation"]
        try:
            tokens, labels = assign_labels_to_tokens(text, spans)
            cleaned_labels = []
            for label in labels:
                label_str = str(label).strip()
                if label_str and not label_str.isdigit() and label_str != "0":
                    cleaned_labels.append(label_str)
            all_labels.extend(cleaned_labels)
            for label in cleaned_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        except Exception as e:
            continue

    min_samples = 3
    filtered_labels = []
    for label in set(all_labels):
        if label_counts.get(label, 0) >= min_samples and label not in ["0", ""]:
            filtered_labels.append(label)

    if "O" not in filtered_labels and "O" in all_labels:
        filtered_labels.append("O")

    label_encoder = LabelEncoder()
    label_encoder.fit(filtered_labels)

    print(f"Классы: {label_encoder.classes_}")
    print(f"Количество классов: {len(label_encoder.classes_)}")

    analyze_labels(synthetic_train_df, label_encoder, "train")
    analyze_labels(synthetic_dev_df, label_encoder, "dev")
    analyze_labels(train_df, label_encoder, "train")
    analyze_labels(dev_df, label_encoder, "dev")

    print("\n4. Создание DataLoader...")
    synthetic_train_dataset = BERTNERDataset(synthetic_train_df, tokenizer, label_encoder, config.max_length)
    synthetic_dev_dataset = BERTNERDataset(synthetic_dev_df, tokenizer, label_encoder, config.max_length)
    train_dataset = BERTNERDataset(train_df, tokenizer, label_encoder, config.max_length)
    dev_dataset = BERTNERDataset(dev_df, tokenizer, label_encoder, config.max_length)

    synthetic_train_loader = DataLoader(synthetic_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    synthetic_dev_loader = DataLoader(synthetic_dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print("\n5. Инициализация модели...")
    model = BERTForNER(
        model_name=config.model_name,
        num_labels=len(label_encoder.classes_),
        hidden_dropout=config.hidden_dropout
    ).to(config.device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_steps = len(synthetic_train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Модель создана: {sum(p.numel() for p in model.parameters()):,} параметров")

    print("\n6. Начало обучения...")
    best_f1 = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        train_loss = train_epoch(model, synthetic_train_loader, optimizer, scheduler, config.device)
        dev_loss, dev_acc, dev_precision, dev_recall, dev_f1 = evaluate(
            model, synthetic_dev_loader, config.device, label_encoder
        )

        print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
        print(f"Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_rubert_model.pth")
            print("🎯 Новый лучший результат! Модель сохранена.")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"⏹️ Ранняя остановка после {epoch + 1} эпох")
                break

    print("\n6. Начало обучения...")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    best_f1 = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.device)
        dev_loss, dev_acc, dev_precision, dev_recall, dev_f1 = evaluate(
            model, dev_loader, config.device, label_encoder
        )

        print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
        print(f"Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_rubert_model.pth")
            print("Новый лучший результат! Модель сохранена.")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Ранняя остановка после {epoch+1} эпох")
                break

    if os.path.exists("best_rubert_model.pth"):
        model.load_state_dict(torch.load("best_rubert_model.pth"))
    print(f"\nЛучшая модель с F1: {best_f1:.4f}")

    print("\n7. ценка на dev...")
    dev_loss, dev_acc, dev_precision, dev_recall, dev_f1 = evaluate(
        model, dev_loader, config.device, label_encoder
    )

    print("\n" + "=" * 50)
    print("МЕТРИКИ НА DEV:")
    print("=" * 50)
    print(f"Accuracy : {dev_acc:.4f}")
    print(f"Precision: {dev_precision:.4f}")
    print(f"Recall   : {dev_recall:.4f}")
    print(f"F1-score : {dev_f1:.4f}")

    print("\n8. Обработка тестового набора...")
    test_df = load_data("../submission.csv", sep=";")
    analyze_labels(test_df, label_encoder, "test")

    test_predictions = predict_on_test_set(
        test_df, model, tokenizer, label_encoder, config.max_length, config.device
    )

    new_df = pd.DataFrame({
        "id": test_predictions.index,
        "search_query": test_predictions["sample"],
        "annotation": test_predictions["annotation"]
    })

    new_df.to_csv("submission.csv", index=False, sep=";")
    print("Предсказания сохранены в test_predictions.csv")

    print("\n9. Сохранение артефактов...")
    torch.save(model.state_dict(), "rubert_ner_model.pth")
    tokenizer.save_pretrained("rubert_tokenizer")
    joblib.dump(label_encoder, "label_encoder_rubert.pkl")

    config_dict = {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "num_classes": len(label_encoder.classes_),
        "hidden_dropout": config.hidden_dropout
    }
    joblib.dump(config_dict, "rubert_config.pkl")

    print("Артефакты сохранены:")
    print("  - rubert_ner_model.pth (веса модели)")
    print("  - rubert_tokenizer/ (токенизатор)")
    print("  - label_encoder_rubert.pkl (кодировщик меток)")
    print("  - rubert_config.pkl (конфигурация)")
    print("  - test_predictions_rubert.csv (предсказания)")

    print("\n=== Обучение завершено ===")

if __name__ == "__main__":
    main()