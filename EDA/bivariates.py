import plotly as py
import plotly.express as px


def plot_bivariate(df, x, y):
    fig = px.scatter(df, x=x, y=y, title=f'{x} vs {y}')
    fig.show()