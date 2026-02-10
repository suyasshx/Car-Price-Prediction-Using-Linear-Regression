import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
data = pd.read_csv(r"D:\internship\pro1\car_data.csv")
data['Age'] = 2025 - data['Year']
data.drop(['Car_Name', 'Year'], axis=1, inplace=True)
data.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
data = pd.get_dummies(data, drop_first=True)
data = data[(data['Selling_Price'] >= (Q1 := data['Selling_Price'].quantile(0.25)) - 1.5 * (Q3 := data['Selling_Price'].quantile(0.75)))]
X, y = data.drop('Selling_Price', axis=1), data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
model = LinearRegression().fit(X_train, y_train)

# Prediction function
def predict_price():
    try:
        inputs = pd.DataFrame([{
            'Present_Price': float(entries["Present Price (in ₹):"].get()),
            'Kms_Driven': int(entries["Kilometers Driven:"].get()),
            'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[entries["Fuel Type:"].get()],
            'Owner': int(entries["Number of Previous Owners:"].get()),
            'Age': int(entries["Age of the Car (in years):"].get()),
            'Seller_Type_Individual': int(entries["Seller Type:"].get() == 'Individual'),
            'Transmission_Manual': int(entries["Transmission Type:"].get() == 'Manual')
        }])
        pred_price = max(0, model.predict(scaler.transform(inputs))[0])
        messagebox.showinfo("Prediction", f"Predicted Selling Price: \u20b9{pred_price:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI setup
app = tk.Tk()
app.title("Car Price Prediction")
app.geometry("400x500")
fields = [
    ("Present Price (in ₹):", "entry"),
    ("Kilometers Driven:", "entry"),
    ("Fuel Type:", ["Petrol", "Diesel", "CNG"]),
    ("Seller Type:", ["Dealer", "Individual"]),
    ("Transmission Type:", ["Manual", "Automatic"]),
    ("Number of Previous Owners:", "entry"),
    ("Age of the Car (in years):", "entry")
]
entries = {label: ttk.Combobox(app, values=widget) if isinstance(widget, list) else tk.Entry(app) for label, widget in fields}
for label, widget in entries.items():
    tk.Label(app, text=label).pack()
    widget.pack()
tk.Button(app, text="Predict Price", command=predict_price, bg="green", fg="white").pack(pady=20)
app.mainloop()
