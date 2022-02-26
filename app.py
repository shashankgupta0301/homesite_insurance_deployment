from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Root URL
@app.route('/')
def index():
    # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

    os.chdir("static/files")
    RFE_columns = pd.read_csv('RFE_features_1.csv').columns

    # Load the test set
    df_test = pd.read_csv('sample.csv')
    RFE_columns = [col for col in RFE_columns if col not in 'QuoteConversion_Flag']  # Test wont have the label column
    df_test = df_test[RFE_columns]
    df_test.drop(columns=['Original_Quote_Date', 'SalesField8'], axis=1, inplace=True)
    # Predict on it using the GBM2 pipeline . Here pipe line has made a task easy as we do not have to store features
    #    y_test = GBM2.predict_proba(df_test)
    loaded_model = pickle.load(open('homesite_prediction_model.pkl', 'rb'))
    y_test = loaded_model.predict_proba(df_test)
    print(y_test)
    print(y_test[:, 1])
    output = round(y_test[:, 1][0], 0)
    print(output)
    if output > 0.5:
        prediction = "Congratulations ! Customer will most probably buy Home Site Insurance."
    else:
        prediction = "Oh No ! Customer will not buy Home Site Insurance."

    return jsonify({'prediction': prediction})
    return redirect(url_for('index'))


if (__name__ == "__main__"):
    app.run(port=5000, debug=True)
