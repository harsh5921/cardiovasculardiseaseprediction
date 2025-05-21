from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.drop(columns=["id"])
    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[DEBUG] X_train_scaled sample:", X_train_scaled[0])  # ✅ confirm scaling

    return X_train_scaled, X_test_scaled, y_train, y_test
