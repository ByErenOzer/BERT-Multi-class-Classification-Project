from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

def calculate_confusion_matrix(predictions, true_labels):
    preds = np.argmax(predictions, axis=1)
    return confusion_matrix(true_labels, preds)

def calculate_classification_report_df(predictions, true_labels, id2label):
    preds = np.argmax(predictions, axis=1)
    report = classification_report(true_labels, preds, output_dict=True, target_names=[id2label[i] for i in sorted(id2label)])
    report_df = pd.DataFrame(report).transpose()
    return report_df

def get_classification_report(predictions, true_labels, id2label):
    preds = np.argmax(predictions, axis=1)
    report = classification_report(true_labels, preds, target_names=[id2label[i] for i in sorted(id2label)])
    return report

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(cr, title='Classification Report'):
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title(title)
    plt.show()

def display_classification_report(report_df):
    print(report_df.to_string())
