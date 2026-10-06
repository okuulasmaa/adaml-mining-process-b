from pathlib import Path
import matplotlib.pyplot as plt
from src.utils import get_data_df


def main():

    data_path = Path("./data/MiningProcess_Flotation_Plant_Database.csv")

    # Preprocessing data 
    data_df = get_data_df(data_path)
    data_df = data_df.drop(columns=["% Silica Concentrate"])
    n_variables = len(data_df.columns)

    # Labels for columns
    colLabels=  []
    for i in range(n_variables):
        colLabels.append("f"+str(i+1))

    # Make a table of the first 5 rows of the data
    fig, ax = plt.subplots(figsize=(8, 2.5))
    table = ax.table(cellText=data_df.head().values,colLabels=colLabels, loc="center")
    table.auto_set_column_width(col=list(range(len(data_df.columns))))
    ax.axis("off")
    save_path = Path("figures/df_head.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()
        

if __name__ == "__main__":
    main()
