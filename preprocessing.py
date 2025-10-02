import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import re
from spellchecker import SpellChecker
from pymorphy3 import MorphAnalyzer
import json

nlp = spacy.load('ru_core_news_lg')

def lemmatize(df: pd.DataFrame) -> pd.DataFrame:
    texts = list(df['sample'])
    cleaned_texts = []
    for text in tqdm(texts, 'Очистка текстов от спецсимволов'):
        # Оставляем буквы, цифры, пробелы и ё/Ё
        cleaned_text = re.sub(r"[^a-zA-ZА-Яа-я0-9ёЁ\s]", ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cleaned_texts.append(cleaned_text)

    corrected_texts = []

    for text in tqdm(cleaned_texts, 'Лемматизация'):
        doc = nlp(text)
        lemmatized_tokens = []
        for token in doc:
            if token.is_space:
                continue
            if not (token.is_punct or token.is_digit):
                lemmatized_tokens.append(token.lemma_)
            else:
                lemmatized_tokens.append(token.text.lower())
        corrected_texts.append(' '.join(lemmatized_tokens))

    df['sample'] = corrected_texts
    return df


def parse_list(s: str) -> list[tuple[int, int, str]]:
    s = s[1:-1]
    tuples_str = s.split('], ')
    result = []
    for t in tuples_str:
        start, end, token_type = t[1:].split(', ')
        result.append((start, end, token_type))
    return result


def is_cyrillic(s: str) -> bool:
    return bool(re.fullmatch(r'[а-яё]+', s, re.IGNORECASE))


def is_latin(s: str) -> bool:
    return bool(re.fullmatch(r'[a-z]+', s, re.IGNORECASE))


def split_camel_case_like(query: str) -> list[str]:
    parts = re.split(r'([а-яё]+)', query, flags=re.IGNORECASE)
    result = []
    for part in parts:
        if not part:
            continue
        if is_cyrillic(part):
            result.append(part)
        else:
            result.append(part)
    return result


def correct_word(word: str, morph: MorphAnalyzer, spell: SpellChecker) -> str:
    if not is_cyrillic(word):
        return word

    word = re.sub(r'(.)\1{2,}', r'\1\1', word)
    word = re.sub(r'(.)\1{2,}', r'\1', word)

    if morph.word_is_known(word):
        return word

    correction = spell.correction(word)
    if correction:
        if morph.word_is_known(correction):
            return correction
    return word

def split_and_correct(query: str, morph: MorphAnalyzer, spell: SpellChecker) -> str:
    tokens = split_camel_case_like(query.lower())
    corrected_tokens = []
    for token in tokens:
        if is_cyrillic(token):
            if morph.word_is_known(token):
                corrected_tokens.append(token)
                continue

            corrected = correct_word(token, morph, spell)
            if morph.word_is_known(corrected):
                corrected_tokens.append(corrected)
                continue

            best_split = [corrected]
            best_score = 1 if morph.word_is_known(corrected) else 0

            n = len(corrected)
            for i in range(1, n):
                left, right = corrected[:i], corrected[i:]

                if len(left) < 2 or len(right) < 2:
                    continue

                if morph.word_is_known(left) and morph.word_is_known(right):
                    score = 2
                    if score > best_score:
                        best_split = [left, right]
                        best_score = score

            corrected_tokens.extend(best_split)
        else:
            corrected_tokens.append(token)
    return re.sub(r'\s+', ' ', ' '.join(corrected_tokens).strip())


def spellcheck(df: pd.DataFrame) -> pd.DataFrame:
    morph = MorphAnalyzer()
    checker = SpellChecker(language=None)

    frequency_df = pd.read_csv('freqrnc2011.csv', sep=';', encoding='1251')
    frequency_df = frequency_df.sort_values(['Lemma', 'Freq(ipm)'])
    lemmas, ipm = list(frequency_df['Lemma']), list(frequency_df['Freq(ipm)'])
    russian_words = {}
    for lemma, freq in zip(lemmas, ipm):
        russian_words[lemma] = int(freq * 10)
    with open("frequencies.json", "w", encoding="utf-8") as f:
        json.dump(russian_words, f, ensure_ascii=False, indent=2)

    checker.word_frequency.load_json("frequencies.json")

    texts = list(df['sample'])
    corrected_texts = [split_and_correct(text, morph, checker) for text in tqdm(texts, 'Spellcheck')]
    df['sample'] = corrected_texts

    return df


def transform_data_to_spacy_format(df: pd.DataFrame) -> DocBin:
    db = DocBin()

    for i in tqdm(df.index, 'Преобразование в формат SpaCy'):
        try:
            row = df.loc[i]
            text, annotation = row['sample'], row['annotation']
            doc = nlp.make_doc(text)
            ents = []
            for t in parse_list(annotation):
                start, end, label = int(t[0]), int(t[1]), t[2]
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        except:
            continue

    return db


df = pd.read_csv('../train.csv', sep=';', encoding='utf-8')
df = lemmatize(df)
df.to_csv('train_data_lemmatized.csv')
df = spellcheck(df)
df.to_csv('train_data_corrected.csv')
db = transform_data_to_spacy_format(df)
db.to_disk('train_data_corrected.spacy')
