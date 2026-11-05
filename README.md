# Spam Detection Lab

Web app that runs the Perceptron vs MLP spam-detection experiment when you click a button, then shows metrics, charts and a "classify your own e-mail" box.

## Run

&#x20;   pip install -r requirements.txt
    python app.py

Open http://127.0.0.1:5000 and press **Run experiment**.

The Enron-Spam dataset (\~15 MB zip) downloads automatically on the first run (or put enron\_spam\_data.csv next to app.py).

## Files

* app.py: Flask backend (trains models in a background thread; /api/run, /api/status, /api/result, /api/predict)
* templates/index.html: single-page frontend, no external libraries (charts are drawn by Matplotlib on the server)

