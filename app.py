"""
Spam Email Detection - Perceptron vs Multilayer Neural Networks
Web app: click a button, the experiment runs on the server, results come back as
metrics + charts. Run:  pip install -r requirements.txt && python app.py
then open http://127.0.0.1:5000
"""
import base64, io, os, re, threading, time, traceback, urllib.request, zipfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "enron_spam_data.csv")
DATA_URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"

app = Flask(__name__)
STATE = {"status": "idle", "progress": 0, "log": [], "result": None, "error": None, "models": None}
LOCK = threading.Lock()


def log(msg, pct=None):
    with LOCK:
        STATE["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if pct is not None:
            STATE["progress"] = pct


def ensure_dataset():
    if os.path.exists(CSV):
        return
    log("Dataset not found locally - downloading Enron-Spam from GitHub (~15 MB)...")
    zpath = os.path.join(HERE, "enron.zip")
    urllib.request.urlretrieve(DATA_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(HERE)
    os.remove(zpath)


def clean(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " url ", t)
    t = re.sub(r"\S+@\S+", " emailaddr ", t)
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_experiment(p):
    seed = int(p["seed"])
    ensure_dataset()
    log("Loading dataset...", 5)
    raw = pd.read_csv(CSV)
    info = {"raw_rows": len(raw)}
    df = raw.copy()
    df["text"] = (df["Subject"].fillna("") + " " + df["Message"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 0]
    info["after_empty"] = len(df)
    df = df.drop_duplicates(subset="text")
    info["after_dedup"] = len(df)
    df["y"] = (df["Spam/Ham"] == "spam").astype(int)
    info["spam"], info["ham"] = int(df.y.sum()), int((1 - df.y).sum())
    log(f"Cleaned: {info['raw_rows']} -> {info['after_dedup']} unique e-mails "
        f"({info['spam']} spam / {info['ham']} ham)", 12)

    log("Normalising text...", 15)
    df["clean"] = df["text"].map(clean)
    info["avg_words"] = round(float(df["clean"].str.split().str.len().mean()), 1)

    Xtr_t, Xte_t, ytr, yte = train_test_split(df["clean"], df["y"], test_size=float(p["test_size"]),
                                              stratify=df["y"], random_state=seed)
    info["train"], info["test"] = len(ytr), len(yte)
    log(f"Split: {len(ytr)} train / {len(yte)} test", 25)

    log("Building TF-IDF features...", 28)
    vec = TfidfVectorizer(max_features=int(p["max_features"]), ngram_range=(1, 2), min_df=3,
                          sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform(Xtr_t)
    Xte = vec.transform(Xte_t)
    info["features"] = Xtr.shape[1]
    log(f"TF-IDF matrix: {Xtr.shape[0]} x {Xtr.shape[1]}", 40)

    mlp = dict(activation="relu", solver="adam", alpha=1e-4, batch_size=128, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=5, max_iter=50, random_state=seed)
    models = {
        "Perceptron": Perceptron(max_iter=1000, tol=1e-3, random_state=seed),
        "MLP (128)": MLPClassifier(hidden_layer_sizes=(128,), **mlp),
        "MLP (128-64)": MLPClassifier(hidden_layer_sizes=(128, 64), **mlp),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
    }
    res, scores, cms = {}, {}, {}
    for i, (n, m) in enumerate(models.items()):
        log(f"Training {n}...", 40 + i * 9)
        t0 = time.time(); m.fit(Xtr, ytr); tt = time.time() - t0
        pred = m.predict(Xte)
        s = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
        cm = confusion_matrix(yte, pred)
        res[n] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                      recall=recall_score(yte, pred), f1=f1_score(yte, pred), auc=roc_auc_score(yte, s),
                      train_time=tt, tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]),
                      epochs=int(np.max(m.n_iter_)) if hasattr(m, "n_iter_") else None)
        scores[n], cms[n] = s, cm
        log(f"  {n}: accuracy {res[n]['accuracy']*100:.2f}%  F1 {res[n]['f1']:.4f}  ({tt:.2f}s)")

    cv = {}
    if p.get("cv"):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        Xall, yall = vec.transform(df["clean"]), df["y"]
        for j, n in enumerate(["Perceptron", "MLP (128)", "MLP (128-64)"]):
            log(f"5-fold cross-validation: {n}...", 85 + j * 4)
            sc = cross_val_score(clone(models[n]), Xall, yall, cv=skf, scoring="f1")
            cv[n] = dict(mean=float(sc.mean()), std=float(sc.std()), folds=[float(x) for x in sc])

    log("Drawing charts...", 97)
    names = list(models)
    plt.rcParams.update({"font.size": 10})
    figs = {}
    # confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for a, n in zip(ax, names[:3]):
        cm = cms[n]; a.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                a.text(c, r, cm[r, c], ha="center", va="center", fontsize=13,
                       color="white" if cm[r, c] > cm.max() / 2 else "black")
        a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_xticklabels(["Ham", "Spam"]); a.set_yticklabels(["Ham", "Spam"])
        a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(n, fontsize=10)
    fig.tight_layout(); figs["confusion"] = b64(fig)
    # metric comparison
    fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.arange(len(names)); w = .2
    for i, mt in enumerate(["accuracy", "precision", "recall", "f1"]):
        ax.bar(x + i * w - .3, [res[n][mt] for n in names], w, label=mt.capitalize())
    lo = min(res[n][m] for n in names for m in ["accuracy", "precision", "recall", "f1"])
    ax.set_ylim(max(0, lo - .01), 1); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Score (axis truncated)"); ax.set_title("Performance comparison")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False)
    fig.tight_layout(); figs["comparison"] = b64(fig)
    # ROC
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for n in names:
        f, t, _ = roc_curve(yte, scores[n]); ax.plot(f, t, lw=1.4, label=f"{n} (AUC {res[n]['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.6); ax.set_xlim(0, .2); ax.set_ylim(.8, 1.005)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (zoomed)"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); figs["roc"] = b64(fig)
    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for n in ["MLP (128)", "MLP (128-64)"]:
        ax[0].plot(models[n].loss_curve_, label=n); ax[1].plot(models[n].validation_scores_, label=n)
    ax[0].set_title("Training loss"); ax[1].set_title("Validation accuracy")
    for a in ax: a.set_xlabel("Epoch"); a.legend(fontsize=8)
    fig.tight_layout(); figs["curves"] = b64(fig)
    # class distribution
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.bar(["Ham", "Spam"], [info["ham"], info["spam"]], color=["#4c78a8", "#e45756"])
    for i, v in enumerate([info["ham"], info["spam"]]): ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Class distribution"); fig.tight_layout(); figs["classdist"] = b64(fig)
    # training time
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.barh(names, [res[n]["train_time"] for n in names], color="#59a14f"); ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)"); ax.set_title("Training cost")
    fig.tight_layout(); figs["time"] = b64(fig)
    # top perceptron features
    fn = np.array(vec.get_feature_names_out()); wts = models["Perceptron"].coef_[0]
    top_s, top_h = np.argsort(-wts)[:12], np.argsort(wts)[:12]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].barh(fn[top_s][::-1], wts[top_s][::-1], color="#e45756"); ax[0].set_title("Strongest spam indicators")
    ax[1].barh(fn[top_h][::-1], -wts[top_h][::-1], color="#4c78a8"); ax[1].set_title("Strongest ham indicators")
    for a in ax: a.set_xlabel("|weight|")
    fig.tight_layout(); figs["features"] = b64(fig)

    with LOCK:
        STATE["models"] = (vec, models)
    return dict(info=info, res=res, cv=cv, figs=figs, params=p,
                top_spam=fn[top_s].tolist(), top_ham=fn[top_h].tolist())


def worker(p):
    try:
        with LOCK:
            STATE.update(status="running", progress=1, log=[], result=None, error=None)
        r = run_experiment(p)
        with LOCK:
            STATE.update(status="done", progress=100, result=r)
        log("Finished.", 100)
    except Exception as e:
        traceback.print_exc()
        with LOCK:
            STATE.update(status="error", error=str(e))
        log(f"ERROR: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    if STATE["status"] == "running":
        return jsonify(error="A run is already in progress"), 409
    d = request.get_json(silent=True) or {}
    p = dict(seed=int(d.get("seed", 42)), test_size=min(max(float(d.get("test_size", .2)), .1), .5),
             max_features=min(max(int(d.get("max_features", 5000)), 500), 20000), cv=bool(d.get("cv", False)))
    threading.Thread(target=worker, args=(p,), daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/status")
def api_status():
    with LOCK:
        return jsonify(status=STATE["status"], progress=STATE["progress"], log=STATE["log"][-60:],
                       error=STATE["error"])


@app.route("/api/result")
def api_result():
    with LOCK:
        if STATE["result"] is None:
            return jsonify(error="No result yet"), 404
        return jsonify(STATE["result"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    with LOCK:
        m = STATE["models"]
    if m is None:
        return jsonify(error="Run the experiment first so the models are trained."), 400
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Enter some text."), 400
    vec, models = m
    X = vec.transform([clean(text)])
    out = {}
    for n, mod in models.items():
        if hasattr(mod, "predict_proba"):
            pr = float(mod.predict_proba(X)[0, 1])
        else:
            pr = float(1 / (1 + np.exp(-mod.decision_function(X)[0])))
        out[n] = dict(label="spam" if mod.predict(X)[0] == 1 else "ham", score=pr)
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
