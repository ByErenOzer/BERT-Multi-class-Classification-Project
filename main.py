import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from model import create_model
from utils import compute_metrics, calculate_confusion_matrix, plot_confusion_matrix, calculate_classification_report_df, plot_classification_report, get_classification_report, display_classification_report
import argparse

# Argümanları ayarla
parser = argparse.ArgumentParser(description="BERT Multi-class Classification")
parser.add_argument("--data_file", type=str, required=True, help="Veri dosyasının yolu")
parser.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-uncased", help="Kullanılacak modelin adı")
parser.add_argument("--output_dir", type=str, default="./FineTuneModel", help="Modelin çıktılarının kaydedileceği dizin")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Eğitim yapılacak epoch sayısı")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Model {device} üzerinde çalışıyor.")

# Veriyi yükle
df_org = pd.read_csv(args.data_file)
df_org = df_org.sample(frac=1.0, random_state=42).reset_index(drop=True)
df_org = df_org.dropna(subset=['clean_text'])

# Etiketleri ve etiket-id haritalarını oluştur
labels = df_org['sentiment'].unique().tolist()
labels = [s.strip() for s in labels]
NUM_LABELS = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

df_org["sentiment"] = df_org.sentiment.map(lambda x: label2id[x.strip()])

# Veriyi train, validation ve test olarak böl
df_train, df_test = train_test_split(df_org, test_size=0.2, random_state=42, stratify=df_org['sentiment'])
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train['sentiment'])

# Tokenizer'ı oluştur
tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=512)
train_encodings = tokenizer(df_train["clean_text"].values.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(df_val["clean_text"].values.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(df_test["clean_text"].values.tolist(), truncation=True, padding=True)

# DataLoader'ları oluştur
train_dataloader = DataLoader(train_encodings, df_train["sentiment"].values.tolist())
val_dataloader = DataLoader(val_encodings, df_val["sentiment"].values.tolist())
test_dataloader = DataLoader(test_encodings, df_test["sentiment"].values.tolist())

# Modeli oluştur ve eğit
model = create_model(args.model_name, NUM_LABELS, id2label, label2id)
model.to(device)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./multi-class-logs',
    logging_steps=250,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_total_limit=1,
    fp16=True,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=False)

# Değerlendirme ve raporlama
eval_results = trainer.evaluate(eval_dataset=test_dataloader)
predictions = trainer.predict(test_dataloader).predictions
true_labels = [item['labels'].item() for item in test_dataloader]

conf_matrix = calculate_confusion_matrix(predictions, true_labels)
class_report_df = calculate_classification_report_df(predictions, true_labels, id2label)
classification_report_text = get_classification_report(predictions, true_labels, id2label)

plot_confusion_matrix(conf_matrix, labels)
plot_classification_report(class_report_df)

# Classification report'u DataFrame olarak yazdır
print("Classification Report DataFrame:")
display_classification_report(class_report_df)

# Modeli kaydet
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
