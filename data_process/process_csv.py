import pandas as pd
import numpy as np
import sys
import os

MIN_CONNECTIONS = 3  #global variable for minimum connections

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py input.csv")
        sys.exit(1)
    
    path = "../dataset/csv/" + sys.argv[1]
    
    df = pd.read_csv(path)
    df = df[['app_id', 'author']].drop_duplicates().reset_index(drop=True)
    
    #filter out entities with less than MIN_CONNECTIONS connections
    while True:
        app_counts = df['app_id'].value_counts()
        author_counts = df['author'].value_counts()
        
        valid_apps = app_counts[app_counts >= MIN_CONNECTIONS].index
        valid_authors = author_counts[author_counts >= MIN_CONNECTIONS].index
        
        prev_size = len(df)
        df = df[df['app_id'].isin(valid_apps) & df['author'].isin(valid_authors)]
        
        if len(df) == prev_size:
            break
    
    #split data (30% train, 70% test)
    total_size = len(df)
    train_size = int(0.3 * total_size)
    test_size = int(0.7 * total_size)
    
    train = df.sample(n=train_size, random_state=42)
    test = df.drop(train.index).sample(n=test_size, random_state=42)
    
    # Create globally unique IDs after splitting
    all_entities = pd.concat([train['app_id'], train['author'], 
                             test['app_id'], test['author']]).unique()
    entity_to_id = {entity: idx for idx, entity in enumerate(all_entities)}
    
    #map to new IDs
    train['app_id_int'] = train['app_id'].map(entity_to_id)
    train['author_id_int'] = train['author'].map(entity_to_id)
    test['app_id_int'] = test['app_id'].map(entity_to_id)
    test['author_id_int'] = test['author'].map(entity_to_id)
    
    # Now offset author IDs to be after all app IDs
    num_apps = len(set(train['app_id']) | set(test['app_id']))
    train['author_id_int'] += num_apps
    test['author_id_int'] += num_apps
    
    #ensure output directory exists
    os.makedirs('../dataset/txt', exist_ok=True)
    
    #save files
    train[['app_id_int', 'author_id_int']].to_csv('../dataset/txt/training.txt', 
                                                sep=' ', index=False, header=False)
    test[['app_id_int', 'author_id_int']].to_csv('../dataset/txt/testing.txt', 
                                               sep=' ', index=False, header=False)
    
    print(f"Total unique apps: {num_apps}")
    print(f"Total unique authors: {len(set(train['author']) | set(test['author']))}")
    print(f"Total nodes in graph: {num_apps + len(set(train['author']) | set(test['author']))}")

if __name__ == '__main__':
    main()
