from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('../data/model.pkl')

@app.route('/', methods=['GET'])
def home():
    html_form = '''
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <title>Predykcja cen domów</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    </head>
    <body>
        <div class="container">
            <h2>Wprowadź dane dla predykcji cen domów</h2>
            <form action="/predict" method="post">
                <label>Crime rate:</label>
                <input type="text" name="CRIM">

                <label>Zoned land %:</label>
                <input type="text" name="ZN">

                <label>Non-retail acres:</label>
                <input type="text" name="INDUS">

                <label>Charles River (0/1):</label>
                <input type="text" name="CHAS">

                <label>Nitric oxides concentration:</label>
                <input type="text" name="NOX">

                <label>Avg rooms per dwelling:</label>
                <input type="text" name="RM">

                <label>Older owner-occupied %:</label>
                <input type="text" name="AGE">

                <label>Distance to employment centers:</label>
                <input type="text" name="DIS">

                <label>Access to radial highways:</label>
                <input type="text" name="RAD">

                <label>Property tax rate:</label>
                <input type="text" name="TAX">

                <label>Pupil-teacher ratio:</label>
                <input type="text" name="PTRATIO">

                <label>Black population (transformed):</label>
                <input type="text" name="B">

                <label>Lower status %:</label>
                <input type="text" name="LSTAT">

                <input type="submit" value="Wyślij">
            </form>
        </div>
    </body>
    </html>
    '''

    return render_template_string(html_form)

# Predykcja
@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                         'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        
        features = [float(request.form.get(name)) for name in feature_names]
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)

        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
