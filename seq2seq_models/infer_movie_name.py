import pandas as pd

file = open('movies.dat')

r_cols = ['movie_id', 'movie_name', 'movie_genre']
metadata = pd.read_csv('movies.dat', sep='::', names=r_cols,
                      encoding='latin-1', index_col='movie_id')
input_list = [2670, 480, 2648, 1340, 2366]
print("Input is: ", )
print(metadata[metadata.index.isin(input_list)])

output_list = ['2859', '3000', '3000', '2686', '224', '224', '224', '2600', '2600', '2961']
print("Output is: ", )
print(metadata[metadata.index.isin(output_list)])