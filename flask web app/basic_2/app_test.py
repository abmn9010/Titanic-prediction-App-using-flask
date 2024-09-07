from flask import Flask, render_template_string
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Load the Iris dataset
    iris = sns.load_dataset('iris')

    # Perform basic EDA
    summary = iris.describe()
    head = iris.head()

    # Create a pairplot
    pairplot = sns.pairplot(iris, hue='species')
    pairplot_file = io.BytesIO()
    plt.savefig(pairplot_file, format='png')
    plt.close()
    pairplot_file.seek(0)
    pairplot_img = base64.b64encode(pairplot_file.getvalue()).decode('utf-8')

    # Render the results in HTML
    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Flask EDA</title>
    </head>
    <body>
        <h1>Iris Dataset EDA</h1>
        <h2>Dataset Summary</h2>
        <pre>{summary.to_html()}</pre>
        <h2>First Few Rows of the Dataset</h2>
        <pre>{head.to_html()}</pre>
        <h2>Pairplot</h2>
        <img src="data:image/png;base64,{pairplot_img}" alt="Pairplot">
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True)
