import argparse
import pandas as pd 

def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except Exception as e:
        print(f"Error loading the dataset: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Load a dataset from a specified file path.")
    parser.add_argument("file_path", help="Path to the dataset file")

    args = parser.parse_args()
    file_path = args.file_path

    dataset = load_dataset(file_path) 
    if dataset is not None:
        print("Dataset loaded successfully.")

        # Data preprocessing
        from dpre import dpre
        dataset = dpre(dataset)

        # Model
        from model import model
        model(dataset)

        # Data visualization
        from vis import vis
        vis(dataset)

        # EDA
        from eda import eda
        eda(dataset)

if __name__ == "__main__":
    main()