import argparse
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, run_repeated_experiment, save_model, preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--save', type=str, default='dtree_model.joblib')
    args = parser.parse_args()

    print("Loading data...")
    df = load_data()

    model = DecisionTreeRegressor(random_state=0)
    mean_mse, mses = run_repeated_experiment(model, df, n_repeats=args.n_repeats, test_size=args.test_size)
    print(f"DecisionTreeRegressor average MSE over {args.n_repeats} runs: {mean_mse:.6f}")
    print("Per-run MSEs:", mses)

    X, y, _ = preprocess(df, scale=False)
    model.fit(X, y)
    save_model(model, args.save)
    print(f"Saved DecisionTreeRegressor to {args.save}")

if __name__ == "__main__":
    main()
