from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin

nlp = spacy.load('ru_core_news_lg')
doc_bin = DocBin().from_disk("train_data_corrected.spacy")

docs = list(doc_bin.get_docs(nlp.vocab))

train_docs, dev_docs = train_test_split(docs, test_size=0.2, random_state=42)

train_bin = DocBin(docs=train_docs)
dev_bin = DocBin(docs=dev_docs)

train_bin.to_disk("train.spacy")
dev_bin.to_disk("dev.spacy")

print(f"Train: {len(train_docs)} документов")
print(f"Dev: {len(dev_docs)} документов")