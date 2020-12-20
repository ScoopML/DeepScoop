import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_losses(x, y):
	sns.set(style="whitegrid")
	data = pd.DataFrame(x, y)
	sns.lineplot(data=data, palette="tab10", linewidth=2.5)
