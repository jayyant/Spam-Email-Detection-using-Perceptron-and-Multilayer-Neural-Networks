# Spam Email Detection using Perceptron and Multilayer Neural Networks

Detects spam e-mail with a single-layer **Perceptron** and **Multilayer Perceptron (MLP)** networks, and measures how much extra depth really helps. Includes a small **web app**: click one button, the experiment runs locally, and you get metrics, charts and a box to classify your own e-mails.

**Authors:** Jayyant Jaju (235891118) and Sumeru Moudgal (235891370)

## Results

Tested on 6,093 unseen e-mails (20% hold-out, seed 42), spam as the positive class:

| Model | Accuracy | F1 | ROC-AUC | Wrong e-mails |
|---|---|---|---|---|
| Perceptron | 98.31% | 0.9823 | 0.9984 | 103 |
| MLP (128) | 98.70% | 0.9864 | 0.9990 | 79 |
| **MLP (128-64)** | **98.82%** | **0.9876** | 0.9988 | **72** |
| Naive Bayes (baseline) | 97.96% | 0.9787 | 0.9975 | 124 |
| Logistic Regression (baseline) | 98.67% | 0.9862 | 0.9987 | 81 |

- Hidden layers cut errors by about 30% versus the perceptron (72 vs 103), but training takes 200+ times longer (about 13 s vs 0.04 s).
- Logistic Regression is almost as accurate, so most of the performance comes from the TF-IDF features, not model complexity.
- Numbers may shift slightly with library versions, seed and hardware.

## How it works

```
Enron-Spam CSV -> clean & de-duplicate -> normalise text -> 80/20 split
               -> TF-IDF (5,000 features) -> train 5 models -> evaluate
```

1. **Clean:** drop empty and duplicate e-mails (33,716 to 30,462) so the same mail can't appear in both train and test sets.
2. **Normalise:** lower-case; links, addresses and numbers become the tokens `url`, `emailaddr`, `num`.
3. **TF-IDF:** turns each e-mail into 5,000 numbers (unigrams + bigrams). Words frequent in one e-mail but rare overall, like "pills", score high. It is fitted on training data only to avoid leakage.
4. **Train:** the Perceptron (one layer, straight-line boundary) and MLPs with one or two hidden layers (ReLU, Adam optimiser, L2 penalty, early stopping), plus Naive Bayes and Logistic Regression as baselines.
5. **Evaluate:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and optional 5-fold cross-validation.

**Web app:** `app.py` (Flask) runs the pipeline in a background thread while `templates/index.html` polls for progress, then shows results. Charts are drawn by Matplotlib on the server, so the page needs no external libraries.

## Quick start

Requires Python 3.9+ (tested on 3.12) and internet on the first run to download the ~15 MB dataset.

```bash
git clone <your-repository-url>
cd <repository-folder>
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** and press **Run experiment**. A run takes about a minute (a few extra minutes with cross-validation). If you are offline, place `enron_spam_data.csv` next to `app.py`.

## Using the app

- **Settings:** test split, number of TF-IDF features, random seed, and optional 5-fold cross-validation.
- **Output:** headline numbers, a comparison table, seven charts (metrics, confusion matrices, ROC, training curves, training time, strongest spam/ham words, class balance) and dataset stats.
- **Try your own e-mail:** paste text and see each model's spam/ham verdict and score. This works only after a run has finished; restarting the server clears the trained models.

No web app needed? Run `mkdir fig && python spam_detection_experiment.py` to print metrics and save `results.json` and `fig/*.png`.

## Project structure

```
app.py                          Flask backend: pipeline + JSON API
templates/index.html            Single-page frontend
requirements.txt                Dependencies
spam_detection_experiment.py    Standalone script (no web app)
docs/                           Project report (.docx)
```

**API:** `POST /api/run` (start; optional `test_size`, `max_features`, `seed`, `cv`), `GET /api/status` (progress and log), `GET /api/result` (metrics and charts), `POST /api/predict` (`{"text": "..."}`).

## Limitations

- All legitimate mail in Enron-Spam comes from Enron staff, so the models partly learn company-specific words. Results on other mailboxes or modern spam are likely lower and untested.
- Text only: headers, attachments, images and HTML are ignored. Hyper-parameters weren't tuned systematically.
- The app uses Flask's development server with in-memory state, so it is for local use, not public deployment.

## Troubleshooting

- **`ModuleNotFoundError`:** activate your virtual environment and re-run `pip install -r requirements.txt`.
- **Port 5000 in use (common on macOS because of AirPlay Receiver):** turn AirPlay Receiver off, or change `port=5000` at the bottom of `app.py`.
- **"Run the experiment first":** the prediction box needs a finished run.

## Dataset and references

Data: [Enron-Spam CSV](https://github.com/MWiechmann/enron_spam_data), built on the Enron corpus (Klimt and Yang, 2004) and the Enron-Spam benchmark (Metsis et al., 2006). Key background: Rosenblatt (1958) on the perceptron, Rumelhart, Hinton and Williams (1986) on back-propagation, and Salton and Buckley (1988) on TF-IDF. The full reference list is in the project report; please verify citations before reuse.

## License

Add your preferred license (e.g. MIT) and a `LICENSE` file. The dataset has its own terms.
