import pandas as pd
import os 

def excel_to_json(excel_file, json_file=None):
    """
    Convert an Excel file to a JSON file.

    Parameters:
    excel_file (str): The path to the Excel file.
    json_file (str, optional): The path to save the JSON file. If not provided, saves in the same directory as the Excel file.
    """
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Save name of the Excel file without extension
    file_name = os.path.splitext(excel_file)[0]
    
    # Convert columns with dates to datetime format 
    df['Geburtsdatum'] = pd.to_datetime(df['Geburtsdatum'], errors='coerce') #Comment out if template not used
    df['Untersuchungsdatum'] = pd.to_datetime(df['Untersuchungsdatum'], errors='coerce') #Comment out if template not used
    
    # Save JSON File and add a number to the file name if it already exists
    if json_file is None:
        json_file = f"{file_name}.json"
    else:
        if not json_file.endswith('.json'):
            json_file += '.json'
        if os.path.exists(json_file):
            base, ext = os.path.splitext(json_file)
            i = 1
            while os.path.exists(json_file):
                json_file = f"{base}_{i}{ext}"
                i += 1
   
    # Convert the DataFrame to a JSON object with indentation
    df.to_json(json_file, orient='records', indent=4, date_format='iso')

    print(f"Excel data has been successfully converted to {json_file}")

# Example usage
# excel_file = 'your_excel_file.xlsx'  # Provide the path to your Excel file
# excel_to_json(excel_file)
