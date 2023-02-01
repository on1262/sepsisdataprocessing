import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
tips.head()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="size", alpha=0.5, linewidth=0, size=0.5)

plt.show()