import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory:", os.getcwd())
# 1
df =pd.read_csv('medical_examination.csv')

# 2
bmi = df['weight'] / np.square(df['height']/100)
df['overweight'] = (bmi > 25).astype('uint8')


# 3
df['gluc']= (df['gluc']>1).astype('uint8')
df['cholesterol'] = (df['cholesterol']>1).astype('uint8')

# 4
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def draw_cat_plot():
    # 5. Convert the data into long format
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6. Draw the catplot
    g = sns.catplot(x="variable", hue="value", col="cardio",
                    data=df_cat, kind="count", height=6, aspect=1.2)

    # 7. Customize the figure as needed (optional)
    g.set_axis_labels("variable", "total")  # Set y-axis label to 'total'
    g.set_titles("cardio = {col_name}")     # Customize the subplot titles
    g.fig.suptitle('Categorical Plot by Cardio', y=1.02)  # Adjust the main title position

    # 8. Save the plot as a file
    fig = g.fig  # Get the figure object

    # 9. Save the figure
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # 11. Calculate the correlation matrix
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()

    # 12. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 14. Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".1f")

    # 15. Customize the figure as needed (optional)
    plt.title('Correlation Heatmap', fontsize=16)

    # 16. Save the figure
    fig.savefig('heatmap.png')
    return fig
