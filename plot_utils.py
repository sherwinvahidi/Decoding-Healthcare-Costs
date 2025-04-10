import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def set_style():
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
    rgb_tuples = [(6, 75, 117), (173, 138, 46), (12, 35, 51), (218, 216, 217), 
              (101, 190, 234), (255, 200, 45), (113, 113, 113),
              (10, 84, 131), (221, 180, 38), (218, 216, 217),
                (12, 125, 193), (246, 174, 28), (160, 158, 159)]
    colors = [mcolors.to_hex([x/255 for x in rgb]) for rgb in rgb_tuples]
    sns.set_palette(sns.color_palette(colors), desat=1)
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": (10, 6),
        "axes.titleweight": "bold",
        "axes.labelcolor": "004C75",
        "text.color": "#004C75",
        "axes.edgecolor": "#ccc"
    })