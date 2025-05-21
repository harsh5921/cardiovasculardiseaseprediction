from train import train, MODELS

for algo in MODELS.keys():
    print(f"\n========== Training {algo.upper()} ==========")
    train(algo=algo, output=f"models/{algo}.joblib")
