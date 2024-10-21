# Vehicle-Resale-Value-Prediction-and-Customer-Centric-Vehicle-Recommendation-System
## Aim of the Project :
The primary aim of this project is to develop a machine learning model that accurately predicts the resale value of vehicles based on various features such as brand, model, age, mileage, and market trends. This project seeks to assist users in making informed decisions regarding the buying and selling of vehicles by providing predictive analytics. Additionally, the project aims to incorporate advanced techniques like collaborative filtering and content-based filtering for personalized vehicle recommendations, as well as integrating virtual voice recognition to enhance user interaction and accessibility.
## Architecture Diagram :

![image](https://github.com/user-attachments/assets/c79eb341-d91e-4567-b608-e04861f4358f)

The graphic provides a concise and clear depiction of all the entities integrated into the Car Resale Value Prediction and Personalized Vehicle Recommendation system. It illustrates how various actions and decisions are interconnected, offering a visual representation of the entire process flow. The diagram  outlines the functional relationships between different entities within the system.
The system architecture shown is clearly demonstrates that the input is provided by the customer in the form of vehicle preferences such as brand, model, and mileage. The system retrieves historical vehicle data from the database, which is then processed through a Random Forest-based prediction model to estimate the resale value of the car. Simultaneously, the depreciation rate is calculated based on the car’s features and market trends.

## Components:
### Hard ware components :
```

●	Processor	:Multi-core CPU (Intel i5/i7 or AMD Ryzen 5/7)
●	Storage         : SSD (at least 256 GB)
●	GPU             : Optional (NVIDIA GTX 1660 or RTX 2060 for deep learning)
●	RAM             : Minimum 8 GB (16 GB preferred)
●	Keyboard        :110 keys enhanced

```
### Software components :
```
•	Operating System                   : Windows, Linux (Ubuntu), or macOS
•	 Programming Language              : Python
•	Machine Learning Libraries         : Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
•	 Development Environment           : Jupyter Notebook, Anaconda, PyCharm, or VS Code
•	 Vesion control                    : Git,Git hub
```
### Technologies Used :
```
•  IDE                                   : Google Colab
•  Programming Language                  : Python
•  Machine Learning (ML)                 : Utilized for predictive modeling and recommendations
•  Data Science (DS)                     : Applied for data analysis and processing
•  Matplotlib                            : Used for data visualization to represent findings graphically
•  Natural Language Processing (NLP)     : Integrated for voice recognition and user interaction features

```
## Algorithm :
```
1. Data Collection:

Gather historical data on vehicle sales, including features like brand, model, year, mileage, condition, and previous sale prices.
Data Preprocessing:

2. Data Cleaning: Handle missing values, remove duplicates, and correct inconsistencies in the dataset.
Feature Engineering: Create new features from existing data (e.g., calculating the age of the vehicle).
Encoding Categorical Variables: Convert categorical features (e.g., brand, model) into numerical representations using techniques like one-hot encoding or label encoding.
3. Exploratory Data Analysis (EDA):

Analyze the data to uncover patterns and relationships between features and resale values. Utilize visualizations to identify trends, outliers, and correlations.
4. Model Selection:

Evaluate and select machine learning algorithms based on the data characteristics. Algorithms used in this project include:
i. Linear Regression: To establish a baseline model for resale value prediction.
ii. Random Forest Regressor: An ensemble method that improves accuracy through multiple decision trees, effectively capturing non-linear relationships.
iii. Gradient Boosting Regressor: A boosting technique that builds models sequentially, focusing on minimizing errors made by previous models.
iv. Support Vector Regressor (SVR): Effective for high-dimensional data, SVR captures complex patterns in the dataset.

5. Model Training:

Split the dataset into training and testing sets (typically an 80/20 split) to evaluate model performance.
Train the selected models on the training set and optimize hyperparameters using techniques like Grid Search or Random Search.
6. Model Evaluation:

Assess model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared values on the test set.
Compare the performance of different models to select the best-performing algorithm for resale value prediction.

7. Deployment:
Integrate a virtual voice recognition feature to enhance user experience, allowing users to query vehicle information verbally.

8. Personalized Recommendations:

Utilize collaborative filtering and content-based filtering techniques to suggest vehicles to users based on their preferences and historical data, enhancing user engagement.
```
## Program :
```
Developed By : D. Vishnuvardhan reddy
Reference Number : 212221230023
```
```
//Read Data
import pandas as pd
data = pd.read_csv('car_data.csv’)
//Data Preprocessing
data.fillna(method='ffill', inplace=True)
data['Age'] = 2024 - data['Year']
data = pd.get_dummies(data, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission'], drop_first=True)
features = data.drop(columns=['Price
target = data['Price']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features[['Kilometers_Driven', 'Engine']] = scaler.fit_transform(features[['Kilometers_Driven', 'Engine’]])
//Model Training
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor 
from sklearn.ensemble import RandomForestRegressor
 from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
// Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
// Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
// Train the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
//Train the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

//Model Evaluation
# Evaluate the models on the test set
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Predictions and evaluation
y_pred_val = gb_model.predict(X_val)
# Calculate RMSE and R² score
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2 = r2_score(y_val, y_pred_val)
print(f'Validation RMSE: {rmse}')
print(f'Validation R² Score: {r2}')
//Resale value prediction
resale_values = pd.DataFrame({'Car': test_data['Engine'], 'Predicted_Resale_Value': y_pred_test})
# Calculate percentiles
best_cars_threshold = resale_values['Predicted_Resale_Value'].quantile(0.75)

def get_user_input_and_find_vehicle():
    # Get user input (can skip with empty input)
    name = input("Enter car name (or press Enter to skip): ")

    # Convert numeric inputs to float/int if provided, otherwise set to None
    try:
        engine = float(input("Enter maximum engine size in CC (or press Enter to skip): ") or None)
    except ValueError:
        engine = None

    try:
        distance_traveled = float(input("Enter maximum distance traveled in km (or press Enter to skip): ") or None)
    except ValueError:
        distance_traveled = None

    try:
        budget = float(input("Enter maximum budget in lakhs (or press Enter to skip): ") or None)

except ValueError:
        budget = None
    try:
        power = float(input("Enter minimum power in bhp (or press Enter to skip): ") or None)
    except ValueError:
        power = None
    try:
        seats = int(input("Enter number of seats (or press Enter to skip): ") or None)
    except ValueError:
        seats = None
    # Find vehicles based on input criteria
    available_vehicles = find_vehicles(
        train_data=train_data_processed,
        name=name,
        engine=engine,
        distance_traveled=distance_traveled,
        budget=budget,
        power=power,
        seats=seats
    )
    print("\nAvailable Vehicles Matching Your Criteria:")
    print(available_vehicles)
# Call the function to get user input and find vehicles
get_user_input_and_find_vehicle()

```
## Output :
![image](https://github.com/user-attachments/assets/c8c64ebd-21be-4c35-9a00-449a309b841f)
![image](https://github.com/user-attachments/assets/c9432dd5-79e2-44db-8958-7f9a04a44e80)
![image](https://github.com/user-attachments/assets/f4b52aa2-7c4b-4bfb-b8e3-586950c5e6ee)
![image](https://github.com/user-attachments/assets/535272eb-ea5e-4450-8ae9-64f710eb7c9c)
![image](https://github.com/user-attachments/assets/8b45cabe-db15-45c7-b426-7cb41ed27b9a)
![image](https://github.com/user-attachments/assets/11c03433-7347-4f5e-adbe-44d057ef4f3a)
![image](https://github.com/user-attachments/assets/3636d3bd-31e6-41b0-8c4d-57510e9903c6)
![image](https://github.com/user-attachments/assets/b051f2e3-99c8-4389-ba41-854cd78fb7ec)
![image](https://github.com/user-attachments/assets/fbca5b16-31ba-40ca-b221-f9b30097e915)
![image](https://github.com/user-attachments/assets/7a0c8dea-357a-45e2-86b5-0b78315d91f4)
![image](https://github.com/user-attachments/assets/5e0f8fd9-7c1c-4c0d-9a66-79d1b51695d1)
![image](https://github.com/user-attachments/assets/2d2a52b7-9cd3-4be9-8550-12b5df8ba44b)

## Result :
The project successfully developed a machine learning model that predicts vehicle resale values, with the Random Forest Regressor demonstrating the highest accuracy (R² of [0.87]) . Insights gained suggest that age and mileage are significant factors influencing resale value, guiding users in their buying and selling decisions.
