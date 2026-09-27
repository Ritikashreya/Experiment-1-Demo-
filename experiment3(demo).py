import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#1. create mock dataset
data= {
    "Age" : [20,21,None,23,24],
    "Income" :[25000,30000,28000,None,4000],
    "city" : ["kolkata","durgapur","kolkata","asansol","Durgapur"],
    "purchased" : [0,1,1,0,1]
}
df = pd.DataFrame(data)

#separate feature and target
x = df.drop("purchased", axis=1)
y = df["purchased"]

numeric_features = ["Age","Income"]
categorical_features =["city"]

#2.define preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

categorical_transformer = Pipeline(steps= [
    ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

#3. combine using columntransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer,numeric_features),
        ("cat", categorical_transformer , categorical_features)
    ]
)
# apply transformations
X_processed =preprocessor.fit_transform(x)

#4. TRAIN-TEST SPLIT
X_train,X_test,y_train, y_test =train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

#Display results

print("---original dataset---")
print(df)

print("\n---processed feature matrix shape ---")
print(X_processed.shape)

print("\nTraining sample:",X_train.shape[0])
print("Testing samples:",X_test.shape[0])