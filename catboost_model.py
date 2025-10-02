import pandas as pd
import re
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
import ast
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_data(path, sep=','):
    df = pd.read_csv(path, sep=sep)
    df["annotation"] = df["annotation"].apply(ast.literal_eval)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, np.str_) else x)

    return df


train_df = load_data("merged_train_data_corrected.csv")
dev_df = load_data("merged_dev_data_corrected.csv")


def tokenize_with_spans(text):
    if isinstance(text, np.str_):
        text = str(text)
    tokens = []
    for match in re.finditer(r'\S+', text):
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


def extract_features(tokens, idx, original_text=""):
    token = str(tokens[idx]) if isinstance(tokens[idx], np.str_) else tokens[idx]

    if isinstance(original_text, np.str_):
        original_text = str(original_text)

    features = {}

    features["original_text"] = original_text.lower()

    features["token"] = token.lower()
    features["token_isupper"] = token.isupper()
    features["token_istitle"] = token.istitle()
    features["token_isdigit"] = token.isdigit()
    features["token_len"] = len(token)

    features["prefix-2"] = token[:2] if len(token) >= 2 else token
    features["prefix-3"] = token[:3] if len(token) >= 3 else token
    features["suffix-2"] = token[-2:] if len(token) >= 2 else token
    features["suffix-3"] = token[-3:] if len(token) >= 3 else token

    if idx > 0:
        prev = tokens[idx - 1]
        prev_str = str(prev) if isinstance(prev, np.str_) else prev
        features["prev_token"] = prev_str.lower()
        features["prev_istitle"] = prev_str.istitle()
    else:
        features["prev_token"] = "<START>"
        features["prev_istitle"] = False

    if idx < len(tokens) - 1:
        next_t = tokens[idx + 1]
        next_str = str(next_t) if isinstance(next_t, np.str_) else next_t
        features["next_token"] = next_str.lower()
        features["next_istitle"] = next_str.istitle()
    else:
        features["next_token"] = "<END>"
        features["next_istitle"] = False

    return features


def prepare_dataset(df):
    X = []
    y = []
    for _, row in df.iterrows():
        text = row["sample"]
        spans = row["annotation"]
        try:
            tokens, labels = assign_labels_to_tokens(text, spans)
        except Exception as e:
            print(f"Ошибка в строке: {text} — {e}")
            continue
        if len(tokens) != len(labels):
            continue
        for i, label in enumerate(labels):
            label_str = str(label) if isinstance(label, np.str_) else label
            X.append(extract_features(tokens, i, original_text=text))
            y.append(label_str)
    return pd.DataFrame(X).fillna("<UNK>"), y


print("Подготовка train...")
X_train, y_train = prepare_dataset(train_df)
print("Подготовка dev...")
X_dev, y_dev = prepare_dataset(dev_df)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_dev_encoded = label_encoder.transform(y_dev)  # используем тот же encoder

missing_in_train = set(y_dev) - set(y_train)
if missing_in_train:
    print(f"⚠️ В dev есть метки, отсутствующие в train: {missing_in_train}")
    # Можно добавить их в encoder вручную, но лучше исправить данные

categorical_features = [
    "token", "prev_token", "next_token",
    "prefix-2", "prefix-3", "suffix-2", "suffix-3"
]
text_features = ["original_text"]

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    verbose=50,
    random_seed=42,
    cat_features=categorical_features,
    text_features=text_features,
    task_type='GPU'
)

model.fit(X_train, y_train_encoded, eval_set=(X_dev, y_dev_encoded), use_best_model=True)

y_pred_encoded = model.predict(X_dev)
y_pred = label_encoder.inverse_transform(y_pred_encoded.flatten())

acc = accuracy_score(y_dev, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_dev, y_pred, average="macro", zero_division=0
)

print("\n" + "=" * 50)
print("Метрики на dev-выборке (macro-усреднение):")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("=" * 50)


def predict_on_test_set(test_df, model, label_encoder):
    results = []

    for _, row in tqdm(test_df.iterrows(), 'Предсказания на тестовом датасете'):
        text = row["sample"]
        if isinstance(text, np.str_):
            text = str(text)

        tokens_with_pos = tokenize_with_spans(text)
        tokens = [t[0] for t in tokens_with_pos]

        token_features = []
        for i in range(len(tokens)):
            features = extract_features(tokens, i, original_text=text)
            token_features.append(features)

        X_test = pd.DataFrame(token_features).fillna("<UNK>")

        y_pred_encoded = model.predict(X_test)
        predicted_labels = label_encoder.inverse_transform(y_pred_encoded.flatten())

        annotations = []
        for (token, start, end), label in zip(tokens_with_pos, predicted_labels):
            if label != "O":  # Сохраняем только сущности (не "O")
                # Конвертируем в обычные Python типы
                start_int = int(start) if isinstance(start, (np.integer, np.int64)) else start
                end_int = int(end) if isinstance(end, (np.integer, np.int64)) else end
                label_str = str(label) if isinstance(label, np.str_) else label
                annotations.append((start_int, end_int, label_str))

        results.append({
            "sample": text,
            "annotation": annotations
        })

    return pd.DataFrame(results)


def evaluate_on_test_set(test_df, model, label_encoder):
    """
    Оценивает модель на тестовом наборе с использованием оригинальных аннотаций
    """
    X_test_all = []
    y_test_all = []

    for _, row in test_df.iterrows():
        text = row["sample"]
        spans = row["annotation"]

        # Конвертируем в обычную строку если это numpy строка
        if isinstance(text, np.str_):
            text = str(text)

        try:
            tokens, true_labels = assign_labels_to_tokens(text, spans)
        except Exception as e:
            print(f"Ошибка в строке: {text} — {e}")
            continue

        if len(tokens) != len(true_labels):
            continue

        for i, true_label in enumerate(true_labels):
            features = extract_features(tokens, i, original_text=text)
            X_test_all.append(features)
            y_test_all.append(str(true_label) if isinstance(true_label, np.str_) else true_label)

    X_test_df = pd.DataFrame(X_test_all).fillna("<UNK>")

    y_pred_encoded = model.predict(X_test_df)
    y_pred = label_encoder.inverse_transform(y_pred_encoded.flatten())

    return y_test_all, y_pred, X_test_df


print("\nЗагрузка тестового набора...")
test_df = load_data("../submission.csv", sep=';')  # Замените на путь к вашему тестовому файлу

print("Предсказание на тестовом наборе...")
test_predictions = predict_on_test_set(test_df, model, label_encoder)

new_df = pd.DataFrame({
    'id': test_predictions.index,
    'search_query': test_predictions['sample'],
    'annotation': test_predictions['annotation']
})
new_df['id'] = new_df['id'] + 1
new_df.to_csv("submission.csv", index=False, sep=';')
print("Предсказания для тестового набора сохранены в submission.csv")

print("\nОценка метрик на тестовом наборе...")
y_test_true, y_test_pred, X_test = evaluate_on_test_set(test_df, model, label_encoder)

test_acc = accuracy_score(y_test_true, y_test_pred)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    y_test_true, y_test_pred, average="macro", zero_division=0
)

test_precision_per_class, test_recall_per_class, test_f1_per_class, support = precision_recall_fscore_support(
    y_test_true, y_test_pred, average=None, zero_division=0, labels=label_encoder.classes_
)

print("\n" + "=" * 60)
print("МЕТРИКИ НА ТЕСТОВОМ НАБОРЕ")
print("=" * 60)
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")
print("\n" + "-" * 60)
print("Метрики по классам:")
print("-" * 60)
print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
print("-" * 60)
for i, class_name in enumerate(label_encoder.classes_):
    print(
        f"{class_name:<15} {test_precision_per_class[i]:<10.4f} {test_recall_per_class[i]:<10.4f} {test_f1_per_class[i]:<10.4f} {support[i]:<10}")

print("\n" + "=" * 60)
print("ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
print("=" * 60)
print(classification_report(y_test_true, y_test_pred, zero_division=0))

model.save_model("ner_catboost.cbm")
pd.to_pickle(label_encoder, "label_encoder.pkl")
pd.to_pickle(X_train.columns.tolist(), "feature_columns.pkl")

print("\nМодель и артефакты сохранены!")

print("\nПример предсказаний для тестового набора:")
print(test_predictions.head())

print("\n" + "=" * 60)
print("СРАВНЕНИЕ ПРЕДСКАЗАНИЙ С ОРИГИНАЛЬНЫМИ АННОТАЦИЯМИ:")
print("=" * 60)
for i in range(min(3, len(test_df))):
    print(f"\nПример {i + 1}:")
    print(f"Текст: {test_df.iloc[i]['sample'][:100]}...")
    print(f"Оригинальные аннотации: {test_df.iloc[i]['annotation']}")
    print(f"Предсказанные аннотации: {test_predictions.iloc[i]['annotation']}")
