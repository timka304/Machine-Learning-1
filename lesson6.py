import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


iris = pd.read_csv(r"Machine Learning\iris.csv")

species = iris["species"]

print(species.value_counts())