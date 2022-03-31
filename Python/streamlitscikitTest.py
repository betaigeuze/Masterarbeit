import streamlit as st
import altair as alt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

melbourne_data = pd.read_csv("melb_data.csv")
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price

melbourne_features = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "Propertycount",
    "BuildingArea",
    "YearBuilt",
    "Lattitude",
    "Longtitude",
]

X = melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.25)

forest_model = RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs=-1)
forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)
# print(mean_absolute_error(val_y, melb_preds))

# Get numerical feature importances
importances = list(forest_model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [
    (feature, round(importance, 2))
    for feature, importance in zip(melbourne_features, importances)
]
# Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
feature_importances_df = pd.DataFrame(
    sorted(feature_importances, key=lambda x: x[1], reverse=True)
)
# Print out the feature and importances
# [print("Variable: {:20} Importance: {}".format(*pair)) for pair in feature_importances]


st.write(melbourne_data.head())
st.write(feature_importances_df.head())
chart = (
    alt.Chart(feature_importances_df)
    .mark_bar(opacity=0.3)
    .encode(x="1:Q", y=alt.Y("0:N", stack=None, sort="-x"))
)
st.altair_chart(chart, use_container_width=True)
