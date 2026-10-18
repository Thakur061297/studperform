import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "D:/project/AI-Data.csv"
TARGET = "Class"
FEATURES = ["raisedhands", "VisITedResources", "Discussion", "StudentAbsenceDays"]
CLASS_ORDER = ["L", "M", "H"]

# menu key -> (title, column, extra countplot kwargs)
CHART_MENU = {
    1: ("Overall marks-class distribution", None, {}),
    2: ("Marks class by semester", "Semester", {}),
    3: ("Marks class by gender", "gender", {"order": ["M", "F"]}),
    4: ("Marks class by nationality", "NationalITy", {}),
    5: ("Marks class by grade", "GradeID",
        {"order": ["G-02", "G-04", "G-05", "G-06", "G-07",
                   "G-08", "G-09", "G-10", "G-11", "G-12"]}),
    6: ("Marks class by section", "SectionID", {}),
    7: ("Marks class by topic", "Topic", {}),
    8: ("Marks class by stage", "StageID", {}),
    9: ("Marks class by absence days", "StudentAbsenceDays", {}),
}
EXIT_KEY = 10


def banner(text, width=60, char="="):
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def show_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def draw_chart(df, key):
    title, column, extra = CHART_MENU[key]
    print(f"\n>> Preparing: {title} ...")
    time.sleep(0.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    if column is None:
        sns.countplot(x=TARGET, data=df, order=CLASS_ORDER, ax=ax)
    else:
        sns.countplot(x=column, hue=TARGET, data=df,
                      hue_order=CLASS_ORDER, ax=ax, **extra)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def chart_menu_loop(df):
    while True:
        banner("CHART MENU", char="-")
        for key, (title, _, _) in CHART_MENU.items():
            print(f"  [{key:>2}] {title}")
        print(f"  [{EXIT_KEY:>2}] Skip charts and continue")
        raw = input("\nYour pick: ").strip()
        if not raw.isdigit():
            print("Please type a number from the menu.")
            continue
        key = int(raw)
        if key == EXIT_KEY:
            print("Moving on to model training...")
            return
        if key in CHART_MENU:
            draw_chart(df, key)
        else:
            print("That option does not exist.")


def prepare_data(df):
    """Keep only the model features + target, and encode text columns."""
    df = df[FEATURES + [TARGET]].copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    return df, encoders


def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=1),
        "Random Forest": RandomForestClassifier(random_state=1),
        "Perceptron": Perceptron(random_state=1),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "MLP Neural Network": MLPClassifier(activation="logistic",
                                            max_iter=500, random_state=1),
    }


def train_and_report(models, X_tr, X_te, y_tr, y_te, class_names):
    scores = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        scores[name] = accuracy_score(y_te, pred)
        banner(name, char="-")
        print(classification_report(y_te, pred, target_names=class_names))
        print(f"Accuracy: {scores[name] * 100:.2f}%")
    return scores


def print_leaderboard(scores):
    banner("MODEL LEADERBOARD")
    for rank, (name, acc) in enumerate(
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True), 1):
        bar = "#" * int(acc * 40)
        print(f"{rank}. {name:<22} {acc * 100:6.2f}%  {bar}")


def ask_number(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("  Enter a whole number.")


def predict_custom(models, encoders):
    banner("CUSTOM PREDICTION")
    hands = ask_number("Raised hands count: ")
    resources = ask_number("Visited resources count: ")
    discussions = ask_number("Discussion count: ")
    absence_enc = encoders["StudentAbsenceDays"]
    while True:
        absence = input("Absence days (Under-7 / Above-7): ").strip()
        if absence in absence_enc.classes_:
            break
        print("  Please type exactly Under-7 or Above-7.")
    row = np.array([[hands, resources, discussions,
                     absence_enc.transform([absence])[0]]])
    class_enc = encoders[TARGET]
    print("\nPredicted marks class per model:")
    for name, model in models.items():
        label = class_enc.inverse_transform(model.predict(row))[0]
        print(f"  {name:<22} -> {label}")


def main():
    raw = pd.read_csv(DATA_PATH)

    show_heatmap(raw)
    chart_menu_loop(raw)

    data, encoders = prepare_data(raw)
    X = data[FEATURES].values
    y = data[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, shuffle=True, random_state=42)

    models = build_models()
    scores = train_and_report(models, X_tr, X_te, y_tr, y_te,
                              encoders[TARGET].classes_)
    print_leaderboard(scores)

    if input("\nTry a custom student record? (y/n): ").strip().lower() == "y":
        predict_custom(models, encoders)
    print("\nDone. Goodbye!")


if __name__ == "__main__":
    main()
