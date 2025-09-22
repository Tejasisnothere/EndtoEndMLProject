from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application




@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        # Show the form
        return render_template('home.html')
    
    else:
        try:
            # Get data from form
            gender = request.form.get("gender")
            race_ethnicity = request.form.get("race_ethnicity")
            parental_level_of_education = request.form.get("parental_level_of_education")
            lunch = request.form.get("lunch")
            test_preparation_course = request.form.get("test_preparation_course")
            reading_score = int(request.form.get("reading_score"))
            writing_score = int(request.form.get("writing_score"))


            # Create CustomData object
            custom_data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            
            pred_df = custom_data.get_data_as_data_frame()

            # Predict using your pipeline
            pipeline = PredictPipeline()
            predicted_score = pipeline.predict(pred_df)

            # Render the same template with prediction
            return render_template('home.html', prediction=predicted_score)

        except Exception as e:
            return f"An error occurred: {str(e)}"



if __name__=="__main__":
    app.run(host="0.0.0.0")