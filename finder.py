import csv

def find_crop(crop_name):
    try:
        # Open the CSV file
        with open("Crop_recommendation.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)

            # Filter rows based on the `label` column (crop name)
            results = [row for row in reader if row.get("label", "").lower() == crop_name.lower()]

        # If results are found, return the relevant data
        if results:
            row = results[0]
            output = f"Info about '{crop_name}':\n"
            output += f"- Nitrogen Required: {int(round(float(row['N'])))}\n"
            output += f"- Phosphorus Required: {int(round(float(row['P'])))}\n"
            output += f"- Potassium Required: {int(round(float(row['K'])))}\n"
            output += f"- Optimal Temperature: {round(float(row['temperature']), 2)}°C\n"
            output += f"- Optimal Humidity: {round(float(row['humidity']), 2)}%\n"
            output += f"- Optimal pH: {round(float(row['ph']), 2)}\n"
            output += f"- Optimal Rainfall: {round(float(row['rainfall']), 2)} mm\n"
            return output
        else:
            return f"No matching crop found for '{crop_name}'."
    except Exception as e:
        return f"An error occurred: {e}"
