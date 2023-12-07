import numpy as np
import pickle
import streamlit as st

Diabetes_model = pickle.load(open("Diabetes_classifier.sav", "rb"))

# Creating a function for prediction
def diabetes_prediction(input_data):
    # changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshaping the input array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # making prediction
    prediction = Diabetes_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "you are not diabetic"
    else:
        return "you are diabetic"

def main():
    # Title
    st.title("Predict Diabetes")

    # Getting input data from user
    pregnancies = st.text_input("no. of pregnancies")
    glucose = st.text_input("Enter Glucose level")
    blood_pressure = st.text_input("Enter Blood Pressure")
    skin_thickness = st.text_input("Enter Skin Thickness")
    insulin = st.text_input("Enter Insulin")
    BMI = st.text_input("Enter BMI")
    diabetes_pedigree_function = st.text_input("Enter Diabetes Pedigree Function")
    age = st.text_input("Enter Age")

    # Enter code for Diagnosis
    diagnosis = " "

    # Creating a button
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction(
            [pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree_function, age]
        )

    st.success(diagnosis)

    # Sidebar for additional pages
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Select a page", ["Home", "About", "Contact Us"])

    if selected_page == "About":
        st.write("This is a Streamlit app for predicting diabetes.")
        # Add more about information here

    elif selected_page == "Contact Us":
        st.write("Contact us at example@email.com.")
        # Add more contact information or a form here

if __name__ == "__main__":
    main()
