from datasets import load_dataset
import pandas as pd

def load_huggingface_dataset(dataset_name, split="train"):
    """
    Load a dataset from Hugging Face.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face
        split (str): Dataset split to load (default: "train")
        
    Returns:
        pandas.DataFrame: The loaded dataset as a pandas DataFrame
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    
    return df

if __name__ == "__main__":
    # Example usage
    # Replace 'tweet_eval' with your desired dataset name
    dataset_name = "tweet_eval"
    df = load_huggingface_dataset(dataset_name)
    print(f"Loaded dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head()) 