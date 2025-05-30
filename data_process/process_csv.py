import pandas as pd
import numpy as np
import sys

min_threshold = 1

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py input.csv")
        sys.exit(1)
    
    path = "../dataset/csv/" + sys.argv[1]
    
    #read the csv file
    df = pd.read_csv(path)
    
    #drop all other irrelevant columns
    df = df[['app_id', 'author']].drop_duplicates().reset_index(drop=True)
    
    #assign unique int id to each row
    df['app_id_int'] = df['app_id'].astype('category').cat.codes
    df['author_id_int'] = df['author'].astype('category').cat.codes
    
    # Split into training (30%) and testing (70%)
    train = df.sample(frac=0.3, random_state=42)
    test = df.drop(train.index)
    
    #save into txt files
    train[['app_id_int', 'author_id_int']].to_csv('../dataset/txt/training.txt', sep=' ', index=False, header=False)
    test[['app_id_int', 'author_id_int']].to_csv('../dataset/txt/testing.txt', sep=' ', index=False, header=False)
    return
    
if __name__ == '__main__':
    main()
    