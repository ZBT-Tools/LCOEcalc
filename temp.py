import plotly.express as px
fig = px.scatter(px.data.iris(), x="sepal_length", y="sepal_width", color="species")
fig.write_image("figure.png", engine="kaleido")