# import sklearn as sk
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import os.path
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import os, pickle, joblib
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, hamming_loss


nltk.download("stopwords")
stop_words = stopwords.words("english")

MODELDIR = Path("models")
MODELDIR.mkdir(exist_ok=True)

def flatten_if_single(x):
    """Jeśli x jest listą długości 1 – zwróć jej pierwszy element."""
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x

def load_or_train_mlb(train_labels, path=Path("models/mlb_model.pkl"), all_labels=None):
    if path.exists():
        print("✓ MLB loaded")
        return joblib.load(path)
    print("… training MultiLabelBinarizer")
    if all_labels is None:
        all_labels = sorted({lbl for sub in train_labels for lbl in sub})
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit(train_labels)
    joblib.dump(mlb, path)
    return mlb




if __name__ == "__main__":
    notebook_dir = os.path.dirname(os.path.abspath("__file__"))
    test_path = os.path.join(notebook_dir, "data", "DM2023_test_docs.tsv")
    train_path = os.path.join(notebook_dir, "data", "DM2023_training_docs_and_labels.tsv")
    vectorizer_path = os.path.join(notebook_dir, "models", "vectorizer.pkl")
    mlb_path = os.path.join(notebook_dir, "models", "mlb_model.pkl")
    lda_path = os.path.join(notebook_dir, "models", "lda_model.pkl")

    test = pd.read_csv(test_path, 
                        sep="\t", 
                        encoding="latin1", 
                        header=None,
                        names=["Textfile", "Text", "Topics"])
                 
    train_full = pd.read_csv(train_path, 
                        sep="\t", 
                        encoding="latin1", 
                        header=None,
                        names=["Textfile", "Text", "Topics"])


    # Separating topics
    train_full["Topics"] = (
        train_full["Topics"]
        .apply(flatten_if_single)        
        .str.split(r"\s*,\s*")         
    )



    # TRAIN
    val   = train_full.iloc[80_000:].reset_index(drop=True)
    train = train_full.iloc[:80_000].reset_index(drop=True)
    print("Train:", train.shape, " Val:", val.shape)

    vec_path = Path("models/vectorizer.pkl")
    if vec_path.exists():
        print("Vectorizer loaded")
        vectorizer = joblib.load(vec_path)
    else:
        print("training Vectorizer...")
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            max_df=0.9,
            min_df=3,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=100_000
        )
        vectorizer.fit(train["Text"])
        joblib.dump(vectorizer, "models/vectorizer.pkl")


    X_train_topics = vectorizer.transform(train["Text"])
    X_val_topics   = vectorizer.transform(val["Text"])
    X_test_topics  = vectorizer.transform(test["Text"])

    all_topics = sorted({lbl for sub in train_full["Topics"] for lbl in sub})

    mlb = load_or_train_mlb(train["Topics"], all_labels=all_topics)

    y_train = mlb.transform(train["Topics"])
    y_val   = mlb.transform(val["Topics"])

   
    if os.path.exists("models/classifier.pkl"):
        print("Found classifier model!")

        with open("models/classifier.pkl", "rb") as f:
            clf = pickle.load(f)

    else:
        print("We need to train classifier first...")

        lr = LinearSVC(C=1.0, dual=False)

        # 3. Predykcja
        clf = OneVsRestClassifier(lr, n_jobs=-1)
        clf.fit(X_train_topics, y_train)


        margins = clf.decision_function(X_val_topics)

        # Przetestuj wiele progów
        best_f1, best_thr = -1, 0
        for thr in np.linspace(-0.5, 1.0, 30):
            y_pred_bin = (margins > thr).astype(int)
            f1 = f1_score(y_val, y_pred_bin, average="samples")
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        print(f"✅ Najlepszy próg = {best_thr:.3f}, F1 = {best_f1:.4f}")

        with open("models/classifier.pkl", "wb") as f:
            pickle.dump(clf, f)





    print("Validation...")
    # y_pred_bin = clf.predict(X_val_topics)
    y_pred_bin = (clf.decision_function(X_val_topics) > best_thr).astype(int)
    y_pred_labels = mlb.inverse_transform(y_pred_bin)

    predicted_topics_list = [list(labels) for labels in y_pred_labels]
    print(predicted_topics_list[:100])

    val["PredictedTopics"] = predicted_topics_list

    y_val_true_bin = mlb.transform(val["Topics"])
    y_val_pred_bin = mlb.transform(val["PredictedTopics"])


    print("micro-F1 :", f1_score(y_val_true_bin, y_val_pred_bin, average="micro"))
    print("macro-F1 :", f1_score(y_val_true_bin, y_val_pred_bin, average="macro"))
    print("sample-F1 :", f1_score(y_val_true_bin, y_val_pred_bin, average="samples"))
    print("Hamming  :", hamming_loss(y_val_true_bin, y_val_pred_bin))