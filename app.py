from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'binaries/files'
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

    os.chdir("binaries/files")
    RFE_columns = pd.read_csv('RFE_features_1.csv').columns

    df_test = pd.read_csv('sample.csv')
    RFE_columns = [col for col in RFE_columns if col not in 'QuoteConversion_Flag']  # Test wont have the label column
    df_test = df_test[RFE_columns]
    df_test.drop(columns=['Original_Quote_Date', 'SalesField8'], axis=1, inplace=True)

    loaded_model = pickle.load(open('homesite_prediction_model.pkl', 'rb'))
    pred_q = loaded_model.predict_proba(df_test)
    print(pred_q)
    print(pred_q[:, 1])
    output = round(pred_q[:, 1][0], 0)
    print(output)
    if output > 0.5:
        prediction = "Congratulations ! This Quote will most probably convert [Positive]."
    else:
        prediction = "Oh No ! Sorry - This Quote will most probably NOT convert [Negative]."

    if os.path.exists("sample.csv"):
        os.remove("sample.csv")
    else:
        print("sample.csv file does not exist")

    return jsonify({'prediction': prediction})
    return redirect(url_for('index'))

if (__name__ == "__main__"):
    app.run(port=5000, debug=True)
