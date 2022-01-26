"""Make plots of results."""
import plotly.graph_objects as go

data = [
    ("Disc LogR", 3907, 0.817, "square"),
    ("Disc EEGNet", 12274, 0.930, "circle"),
    ("Disc 1D CNN", 542210, 1.103, "cross"),
    ("Disc 2D CNN", 1603042, 1.153, "x"),
    ("Gen LDA", 819401, 0.678, "diamond"),
    ("Gen LogR", 819401, 0.218, "star-triangle-up"),
]

fig = go.Figure()
fig.update_layout(
    template="simple_white",
    # title="ITR vs Parameter Count",
    xaxis_title="Parameter Count",
    yaxis_title="ITR (bits per symbol)",
    legend=dict(
        orientation="v",
        yanchor="auto",
        y=0,
        xanchor="auto",
        x=1,
        borderwidth=2,
    ),
    font=dict(size=20),
    autosize=False,
    height=400,
    width=600,
    margin=dict(l=50, r=10, b=50, t=10, pad=4),
)
for model, params, itr, symbol in data:
    fig.add_trace(go.Scatter(x=[params], y=[itr], mode="markers", marker=dict(size=15, symbol=symbol), name=model))
fig.write_image("itr-vs-params.png")
fig.show()
