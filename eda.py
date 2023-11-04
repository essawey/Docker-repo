def eda(df):
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    columns = df.columns
    results = np.zeros((len(columns), len(columns)))

    for i in range(len(columns)):
        for j in range(len(columns)):
            col1 = df[columns[i]]
            col2 = df[columns[j]]
            similarity = cosine_similarity(col1, col2)
            results[i, j] = similarity


    with open('eda-in-1.txt', 'w') as f:
        for i in range(len(columns)):
            for j in range(len(columns)):
                f.write(f'Cosine similarity between {columns[i]} and {columns[j]}: {results[i, j]}\n')