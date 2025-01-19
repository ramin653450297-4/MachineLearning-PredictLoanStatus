import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk
import numpy as np
from ttkthemes import ThemedTk
from tkinter import messagebox

# Load pre-trained model, scaler, and feature columns
model = joblib.load('Loan_decision_tree.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Extract occupation list from feature columns
all_occupations = [col.replace('occupation_', '') for col in feature_columns if col.startswith('occupation_')]

# GUI prediction function
def predict_loan_status():
    try:
        # Collect input values from GUI
        age = age_entry.get()
        gender = gender_combobox.get()
        income = income_entry.get()
        credit_score = credit_score_entry.get()
        education_level = education_combobox.get()
        marital_status = marital_combobox.get()
        occupation = occupation_combobox.get()

        # Validate inputs
        if not age.isdigit() or not income.replace('.', '', 1).isdigit() or not credit_score.replace('.', '', 1).isdigit():
            messagebox.showerror("Invalid Input", "Please check your numeric inputs.")
            return

        # Convert inputs to appropriate data types
        age = int(age)
        income = float(income)
        credit_score = float(credit_score)

        # Map categorical inputs to numeric values
        gender_map = {'Female': 0, 'Male': 1}
        education_map = {'Bachelor': 0, 'Master': 1, 'High School': 2, 'Associate': 3, 'Doctoral': 4}
        marital_map = {'Single': 0, 'Married': 1}

        gender_value = gender_map.get(gender, -1)
        education_value = education_map.get(education_level, -1)
        marital_value = marital_map.get(marital_status, -1)

        # Validate mapped values
        if gender_value == -1 or education_value == -1 or marital_value == -1:
            messagebox.showerror("Invalid Input", "Please check your inputs.")
            return

        # One-hot encoding for occupation
        occupation_encoded = [1 if occ == occupation else 0 for occ in all_occupations]

        # Prepare feature array
        input_data = pd.DataFrame([[age, gender_value] + occupation_encoded + [education_value, marital_value, income, credit_score]],
                                  columns=feature_columns)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the loaded model
        prediction = model.predict(input_data_scaled)

        # Display prediction result
        if prediction[0] == 1:
            messagebox.showinfo("Loan Status", "Loan Approved")
        else:
            messagebox.showinfo("Loan Status", "Loan Denied")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Create GUI using Tkinter
root = ThemedTk(theme="arc")
root.title("Loan Approval Prediction")
root.geometry("400x500")

# GUI Components
ttk.Label(root, text="Age:").grid(row=0, column=0, padx=10, pady=10)
age_entry = ttk.Entry(root)
age_entry.grid(row=0, column=1, padx=10, pady=10)

ttk.Label(root, text="Gender:").grid(row=1, column=0, padx=10, pady=10)
gender_combobox = ttk.Combobox(root, values=["Female", "Male"])
gender_combobox.grid(row=1, column=1, padx=10, pady=10)

ttk.Label(root, text="Income:").grid(row=2, column=0, padx=10, pady=10)
income_entry = ttk.Entry(root)
income_entry.grid(row=2, column=1, padx=10, pady=10)

ttk.Label(root, text="Credit Score:").grid(row=3, column=0, padx=10, pady=10)
credit_score_entry = ttk.Entry(root)
credit_score_entry.grid(row=3, column=1, padx=10, pady=10)

ttk.Label(root, text="Education Level:").grid(row=4, column=0, padx=10, pady=10)
education_combobox = ttk.Combobox(root, values=["Bachelor", "Master", "High School", "Associate", "Doctoral"])
education_combobox.grid(row=4, column=1, padx=10, pady=10)

ttk.Label(root, text="Marital Status:").grid(row=5, column=0, padx=10, pady=10)
marital_combobox = ttk.Combobox(root, values=["Single", "Married"])
marital_combobox.grid(row=5, column=1, padx=10, pady=10)

ttk.Label(root, text="Occupation:").grid(row=6, column=0, padx=10, pady=10)
occupation_combobox = ttk.Combobox(root, values=all_occupations)
occupation_combobox.grid(row=6, column=1, padx=10, pady=10)

# Predict button
predict_button = ttk.Button(root, text="Predict Loan Status", command=predict_loan_status)
predict_button.grid(row=7, column=0, columnspan=2, pady=20)

# Run the application
root.mainloop()
