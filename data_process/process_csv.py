import pandas as pd
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py input.csv")
        sys.exit(1)
    
    path = "../dataset/csv/" + sys.argv[1]
    
    # Read and prepare data
    df = pd.read_csv(path)
    df = df[['app_id', 'author']].drop_duplicates().reset_index(drop=True)
    
    # Create globally unique IDs
    # Combine all entities and assign unique IDs
    all_entities = pd.concat([df['app_id'], df['author']]).unique()
    entity_to_id = {entity: idx for idx, entity in enumerate(all_entities)}
    
    # Map to new IDs
    df['app_id_int'] = df['app_id'].map(entity_to_id)
    df['author_id_int'] = df['author'].map(entity_to_id)
    
    # Now offset author IDs to be after all app IDs
    num_apps = df['app_id_int'].nunique()
    df['author_id_int'] += num_apps
    
    # Split data (3% train, 7% test)
    total_size = len(df)
    train_size = int(0.03 * total_size)
    test_size = int(0.07 * total_size)
    
    train = df.sample(n=train_size, random_state=42)
    test = df.drop(train.index).sample(n=test_size, random_state=42)
    
    # Ensure output directory exists
    os.makedirs('../dataset/txt', exist_ok=True)
    
    # Save files
    train[['app_id_int', 'author_id_int']].to_csv('../dataset/txt/training.txt', 
                                                sep=' ', index=False, header=False)
    test[['app_id_int', 'author_id_int']].to_csv('../dataset/txt/testing.txt', 
                                               sep=' ', index=False, header=False)
    
    print(f"Total unique apps: {num_apps}")
    print(f"Total unique authors: {df['author_id_int'].nunique()}")
    print(f"Total nodes in graph: {num_apps + df['author_id_int'].nunique()}")

if __name__ == '__main__':
    main()