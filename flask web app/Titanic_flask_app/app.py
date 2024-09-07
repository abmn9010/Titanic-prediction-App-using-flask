from flask import Flask, render_template, request
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Titanic dataset
df = sns.load_dataset('titanic')
df.dropna(subset=['age', 'embarked', 'fare', 'sex'], inplace=True)

# Feature engineering
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Splitting the data
X = df[['sex', 'age', 'fare', 'embarked']]
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Home route
@app.route('/')
def index():
    # Visualization of Age Distribution by Gender
    fig1 = px.histogram(df, x='age', color='sex', title='Age Distribution by Gender')
    
    # Visualization of Fare by Embarked
    fig2 = px.box(df, x='embarked', y='fare', color='sex', 
                  labels={'embarked': 'Embarked (0 = C, 1 = Q, 2 = S)', 'fare': 'Fare'},
                  title='Fare Distribution by Embarkation and Gender')

    # Visualization of Fare vs Age
    fig3 = px.scatter(df, x='age', y='fare', color='sex', 
                      labels={'age': 'Age', 'fare': 'Fare'},
                      title='Fare vs. Age by Gender')
    
    # Render the graphs as HTML
    graph1 = fig1.to_html(full_html=False)
    graph2 = fig2.to_html(full_html=False)
    graph3 = fig3.to_html(full_html=False)
    
    return render_template('index.html', graph1=graph1, graph2=graph2, graph3=graph3)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    embarked = int(request.form['embarked'])

    # Prepare input data
    input_data = scaler.transform([[sex, age, fare, embarked]])
    prediction = model.predict(input_data)
    survival = 'Survived' if prediction[0] == 1 else 'Did not survive'

    return render_template('result.html', survival=survival)

if __name__ == '__main__':
    app.run(debug=True)
