import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

data = {
    'Red': [255, 200, 150, 100, 50],
    'Green': [0, 100, 150, 200, 255],
    'Blue': [0, 50, 100, 150, 255],
    'Color': ['Red', 'Orange', 'Yellow', 'Green', 'Blue']
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Color_Label'] = le.fit_transform(df['Color'])

X = df[['Red', 'Green', 'Blue']]
y = df['Color_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

new_color = {'Red': 120, 'Green': 180, 'Blue': 60}
new_color_df = pd.DataFrame([new_color])
prediction = model.predict(new_color_df)
predicted_color = le.inverse_transform(prediction.astype(int))[0]

print(f'Predicted Color: {predicted_color}')
