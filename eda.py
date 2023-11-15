def eda(df):
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    numeric_df = df.select_dtypes(include=[np.number])
    columns = numeric_df.columns
    results = np.zeros((len(columns), len(columns)))
    
    # Calculate the Cosine similarity
    for i in range(len(columns)):
        for j in range(len(columns)):
            col1 = numeric_df[columns[i]]
            col2 = numeric_df[columns[j]]
            similarity = cosine_similarity(col1, col2)
            results[i, j] = similarity

    with open('eda-in-1.txt', 'w') as f:

        # write the Cosine similarity
        for i in range(len(columns)):
            for j in range(len(columns)):
                f.write(f'Cosine similarity between {columns[i]} and {columns[j]}: {results[i, j]}\n')
        f.write('\n')

        # Calculate the mean and median
        for col in columns:
            mean_val = numeric_df[col].mean()
            median_val = numeric_df[col].median()
            f.write(f'{col} has Mean of: {mean_val} and a Median of {median_val}\n')

        # Calculate the columns correlation
        correlation_matrix = numeric_df.corr()
        f.write('\nCorrelation between different features:\n')
        f.write(correlation_matrix.to_string())

        # Calculate the highest values
        Highest_fare = df['Fare'].max()
        Biggest_family_size = df['FamilySize'].max()

        f.write(f'\n\nHighest Fare: {Highest_fare}\n')
        f.write(f'Biggest Family Size: {Biggest_family_size}\n')