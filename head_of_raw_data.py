import matplotlib.pyplot as plt
import function_storage as fs

def main():

    # Preprocessing data 
    data_df = fs.get_data_df()
    data_df = data_df.drop(columns=["% Silica Concentrate"])
    n_variables = len(data_df.columns)

    # Labels for columns
    colLabels=  []
    for i in range(n_variables):
        colLabels.append("f"+str(i+1))

    # Make a table of the first 5 rows of the data
    fig, ax = plt.subplots(figsize=(8, 2.5))
    table = ax.table(cellText=data_df.head().values,colLabels=colLabels,loc="center")
    table.auto_set_column_width(col=list(range(len(data_df.columns))))
    ax.axis("off")
    plt.savefig("figures/df_head.pdf",bbox_inches="tight",pad_inches=0)
    plt.show()
    plt.close()
        

if __name__ == "__main__":
    main()
