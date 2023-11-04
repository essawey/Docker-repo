def vis(df):
    import missingno as msno
    import matplotlib.pyplot as plt
    msno.matrix(df)
    plt.savefig('vis.png')
