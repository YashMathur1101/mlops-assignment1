import argparse
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, run_repeated_experiment, save_model, preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--save', type=str, default='kernelridge_model.joblib')
    args = parser.parse_args()

    print("Loading data...")
    df = load_data()

    model = KernelRidge(alpha=args.alpha, kernel=args.kernel)
    mean_mse, mses = run_repeated_experiment(model, df, n_repeats=args.n_repeats, test_size=args.test_size)
    print(f"KernelRidge average MSE over {args.n_repeats} runs: {mean_mse:.6f}")
    print("Per-run MSEs:", mses)

    X, y, _ = preprocess(df, scale=False)
    model.fit(X, y)
    save_model(model, args.save)
    print(f"Saved KernelRidge to {args.save}")

if __name__ == "__main__":
    main()
