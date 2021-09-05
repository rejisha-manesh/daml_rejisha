from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('Final_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('StrokeRecovery.jpg')
    image_head = Image.open('STROKE.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict stroke')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_head)
    st.title("Predicting Stroke")
    if add_selectbox == 'Online':
        age=st.number_input('age' , min_value=1, max_value=85, value=1)
        avg_glucose_level =st.number_input('avg_glucose_level',min_value=50, max_value=250, value=50)
        bmi = st.number_input('bmi', min_value=20, max_value=50, value=20)
        heart_disease = st.selectbox('heart_disease', ['0', '1'])
        hypertension = st.selectbox('hypertension', ['0', '1'])
        gender = st.selectbox('gender', ['Male', 'Female'])
        work_type = st.selectbox('work_type', ['Private','Self-employed','Govt_job'])
        ever_married = st.selectbox('ever_married', ['Yes', 'No'])
        smoking_status = st.selectbox('smoking_status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
        Residence_type = st.selectbox('Residence_type', ['Urban', 'Rural'])
        
        output=""
        input_dict={'age':age,'avg_glucose_level':avg_glucose_level,'bmi':bmi,'heart_disease':heart_disease,'hypertension': hypertension,'gender':gender,'work_type' : work_type,'ever_married' : ever_married,'smoking_status' : smoking_status,'Residence_type' : Residence_type}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == "__main__":
  run()
