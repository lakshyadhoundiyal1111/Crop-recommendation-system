from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
import numpy as np
import pandas as pd
import requests

class Cropping:
    def __init__(self, root):
        self.root = root
        self.root.title("CROPPING RECOMMENDATION SYSTEM")
        self.root.geometry("1360x700+0+0")
        self.root.resizable(False, False)

        self.region = StringVar()
        self.crop = StringVar()
        self.soil = StringVar()
        self.name = StringVar()
        self.temperature = StringVar()
        self.humidity = StringVar()
        self.rainfall = StringVar()
        self.id = 0
        self.season = StringVar()
        self.model = None

        # Initialize label encoders and mappers
        self.le_region = LabelEncoder()
        self.le_crop = LabelEncoder()
        self.le_soil = LabelEncoder()
        self.region_mapping = {}
        self.crop_mapping = {}
        self.soil_mapping = {}
        self.scaler = StandardScaler()

        # Heading label
        label1 = Label(self.root, text="CROPPING SEASON RECOMMENDATION SYSTEM", font=("lucida sans", 20, "bold"), bg="#81CAD6", fg="white", relief="solid")
        label1.place(x=2, y=2, width=1354, height=100)

        # Background label
        label2 = Label(self.root, bg="#EDCD44", relief="solid")
        label2.place(x=2, y=104, width=1354, height=590)

        # Label frame1
        labelframe1 = LabelFrame(label2, text="CROPPING DETAILS", relief="solid", bg="white")
        labelframe1.place(x=30, y=30, width=650, height=510)

        # Region label
        label3 = Label(labelframe1, text="REGION", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label3.place(x=30, y=30, width=250, height=40)

        combobox1 = ttk.Combobox(labelframe1, font=("lucida sans", 15, "bold"), state="readonly", textvariable=self.region)
        combobox1["values"] = ["SELECT REGION", "Delhi", "Mumbai", "Rohtak", "Jaisalmer", "Kolkata", "Jaipur"]
        combobox1.current(0)
        combobox1.place(x=330, y=30, width=280, height=40)

        # Crop type label
        label4 = Label(labelframe1, text="CROP", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label4.place(x=30, y=100, width=250, height=40)

        combobox2 = ttk.Combobox(labelframe1, font=("lucida sans", 15, "bold"), state="readonly", textvariable=self.crop)
        combobox2["values"] = ["SELECT CROP", "Rice", "Wheat", "Cotton", "Tea"]
        combobox2.current(0)
        combobox2.place(x=330, y=100, width=280, height=40)

        # Soil type
        label5 = Label(labelframe1, text="SOIL_TYPE", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label5.place(x=30, y=170, width=250, height=40)

        combobox3 = ttk.Combobox(labelframe1, font=("lucida sans", 15, "bold"), state="readonly", textvariable=self.soil)
        combobox3["values"] = ["SELECT SOIL", "Alluvial", "Red", "Black", "Clayey"]
        combobox3.current(0)
        combobox3.place(x=330, y=170, width=280, height=40)

        # Name label
        label6 = Label(labelframe1, text="NAME", font=("lucida sans", 15, "bold"), bg="#DB504A", fg="white")
        label6.place(x=30, y=240, width=250, height=40)

        entry1 = ttk.Entry(labelframe1, font=("lucida sans", 15, "bold"), textvariable=self.name)
        entry1.place(x=330, y=240, width=280, height=40)

        # Cropping season button
        b1 = Button(labelframe1, text="PRESS TO GENERATE CROPPING SEASON", font=("lucida sans", 15, "bold"), cursor="hand2", bg="#084C61", relief="solid", fg="White", command=self.season_generation_button)
        b1.place(x=30, y=350, width=580, height=80)

        # Frame
        f1 = Frame(label2, bg="white")
        f1.place(x=700, y=30, width=620, height=510)

        # Scrollbar
        scroll_x = ttk.Scrollbar(f1, orient="horizontal")
        scroll_y = ttk.Scrollbar(f1, orient="vertical")

        # Treeview
        self.cropping_table = ttk.Treeview(f1, columns=("S.NO.", "FARMER_NAME", "REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "RAINFALL", "HUMIDITY", "SEASON"), xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        scroll_x.pack(side="bottom", fill="x")
        scroll_y.pack(side="right", fill="y")
        scroll_x.config(command=self.cropping_table.xview)
        scroll_y.config(command=self.cropping_table.yview)

        self.cropping_table.heading("S.NO.", text="S.NO.")
        self.cropping_table.heading("FARMER_NAME", text="FARMER_NAME")
        self.cropping_table.heading("REGION", text="REGION")
        self.cropping_table.heading("CROP", text="CROP")
        self.cropping_table.heading("SOIL_TYPE", text="SOIL_TYPE")
        self.cropping_table.heading("TEMPERATURE", text="TEMPERATURE")
        self.cropping_table.heading("RAINFALL", text="RAINFALL")
        self.cropping_table.heading("HUMIDITY", text="HUMIDITY")
        self.cropping_table.heading("SEASON", text="SEASON")

        self.cropping_table["show"] = "headings"

        self.cropping_table.column("S.NO.", width=150)
        self.cropping_table.column("FARMER_NAME", width=150)
        self.cropping_table.column("REGION", width=150)
        self.cropping_table.column("CROP", width=150)
        self.cropping_table.column("SOIL_TYPE", width=150)
        self.cropping_table.column("TEMPERATURE", width=150)
        self.cropping_table.column("RAINFALL", width=150)
        self.cropping_table.column("HUMIDITY", width=150)
        self.cropping_table.column("SEASON", width=150)

        self.cropping_table.pack(fill="both", expand=1)

    def season_generation_button(self):
        if self.region.get() == "SELECT REGION" or self.crop.get() == "SELECT CROP" or self.soil.get() == "SELECT SOIL" or self.name.get() == "":
            messagebox.showerror("ERROR", "All fields should be filled!!!", parent=self.root)
        else:
            self.temperature, self.rainfall, self.humidity = self.get_weather_data()

            if self.temperature is None or self.rainfall is None or self.humidity is None:
                messagebox.showerror("Error", "Failed to retrieve weather data.")
                return

            # Create the input feature set as a DataFrame
            x_input = pd.DataFrame([[self.region.get(), self.crop.get(), self.soil.get(), self.temperature, self.humidity, self.rainfall]], 
                                   columns=["REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "HUMIDITY", "RAINFALL"])

            # Load data and train the model
            x, y, le_region, le_crop, le_soil, scaler = self.load_data_and_train(x_input)

            if x is None:
                return

            # Encode the input data using label encoders
            try:
                x_input["REGION"] = le_region.transform([self.region.get()])
                x_input["CROP"] = le_crop.transform([self.crop.get()])
                x_input["SOIL_TYPE"] = le_soil.transform([self.soil.get()])
            except ValueError as e:
                messagebox.showerror("Error", f"Encoding error: {e}")
                return

            # Scale the input data
            x_input = scaler.transform(x_input)

            # Predict the season
            self.season = self.predict_season(self.model, x_input)

            if self.season == "Unknown":
                messagebox.showerror("Error", "Failed to predict the cropping season.")
                return

            conn = mysql.connector.connect(host="localhost", user="root", password="Amar0311@0704", database="cropping_recommendation_system")
            my_cursor = conn.cursor()
            my_cursor.execute("INSERT INTO recommendation (ID,FARMER_NAME, REGION, CROP, SOIL_TYPE, TEMPERATURE, RAINFALL, HUMIDITY, SEASON) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                              (self.id, self.name.get(), self.region.get(), self.crop.get(), self.soil.get(), self.temperature, self.rainfall, self.humidity, self.season))
            self.id += 1
            conn.commit()
            conn.close()

            self.fetch_data()

            messagebox.showinfo("Prediction", f"The recommended cropping season is: {self.season}")

    def get_weather_data(self):
        try:
            # Simulate getting weather data (temperature, rainfall, humidity)
            temperature = np.random.uniform(20, 40)
            rainfall = np.random.uniform(50, 300)
            humidity = np.random.uniform(40, 90)
            return temperature, rainfall, humidity
        except Exception as e:
            print(f"Error in getting weather data: {e}")
            return None, None, None

    def load_data_and_train(self, x_input):
        try:
            data = pd.DataFrame({
                "REGION": ["Delhi", "Mumbai", "Rohtak", "Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Lucknow", "Ahmedabad",
                        "Pune", "Jaipur", "Chandigarh", "Patna", "Hyderabad", "Bhopal", "Indore", "Nagpur", "Vadodara", "Surat"],
                "CROP": ["Rice", "Wheat", "Cotton", "Tea", "Rice", "Sugarcane", "Maize", "Barley", "Jute", "Pulses",
                        "Sorghum", "Millets", "Groundnut", "Mustard", "Sunflower", "Soybean", "Tobacco", "Coffee", "Pepper", "Cardamom"],
                "SOIL_TYPE": ["Alluvial", "Red", "Black", "Clayey", "Alluvial", "Alluvial", "Alluvial", "Loamy", "Alluvial", "Black",
                            "Red", "Sandy", "Sandy", "Loamy", "Red", "Black", "Clayey", "Clayey", "Red", "Loamy"],
                "TEMPERATURE": [30, 25, 35, 20, 33, 32, 28, 24, 30, 27, 26, 29, 31, 20, 32, 27, 24, 23, 28, 26],
                "HUMIDITY": [60, 70, 80, 50, 55, 65, 75, 55, 85, 68, 72, 63, 58, 70, 67, 75, 60, 78, 62, 65],
                "RAINFALL": [100, 150, 200, 90, 110, 200, 180, 75, 240, 130, 140, 160, 110, 120, 170, 125, 155, 190, 210, 180],
                "SEASON": ["June - October", "November - March", "June - October", "November - March", "June - October", "June - October", 
                        "June - October", "November - March", "June - October", "November - March", "June - October", 
                        "June - October", "June - October", "November - March", "April - May", "June - October", "November - March", 
                        "April - May", "June - October", "April - May"]
            })

            x = data[["REGION", "CROP", "SOIL_TYPE", "TEMPERATURE", "HUMIDITY", "RAINFALL"]]
            y = data["SEASON"]

            # Initialize label encoders
            le_region = LabelEncoder()
            le_crop = LabelEncoder()
            le_soil = LabelEncoder()

            # Fit and transform label encoders
            x["REGION"] = le_region.fit_transform(x["REGION"])
            x["CROP"] = le_crop.fit_transform(x["CROP"])
            x["SOIL_TYPE"] = le_soil.fit_transform(x["SOIL_TYPE"])

            # Standardize the data
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

            # Split the data into train and test sets
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Train the model
            self.model = RandomForestClassifier()
            self.model.fit(x_train, y_train)

            # Evaluate the model
            y_pred = self.model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy * 100:.2f}%")

            return x, y, le_region, le_crop, le_soil, scaler

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data and train the model: {str(e)}")
            return None, None, None, None, None, None

    def predict_season(self, model, x_input):
        if model is not None:
            prediction = model.predict(x_input)
            season = prediction[0]
            print(f"Predicted season: {season}")
            return season
        else:
            print("Model is not trained or available.")
            return "Unknown"

    def fetch_data(self):
        conn = mysql.connector.connect(host="localhost", user="root", password="Amar0311@0704", database="cropping_recommendation_system")
        my_cursor = conn.cursor()
        my_cursor.execute("SELECT * FROM recommendation")
        rows = my_cursor.fetchall()
        if len(rows) != 0:
            self.cropping_table.delete(*self.cropping_table.get_children())
            for row in rows:
                self.cropping_table.insert("", END, values=row)
            conn.commit()
        conn.close()

if __name__ == "__main__":
    root = Tk()
    obj = Cropping(root)
    root.mainloop()
