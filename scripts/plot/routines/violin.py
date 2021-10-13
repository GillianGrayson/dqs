import plotly.graph_objects as go


def add_violin_trace(fig, y, name, x=None, showlegend=True):

    if x is not None:
        fig.add_trace(
            go.Violin(
                x=x,
                y=y,
                name=name,
                box_visible=True,
                showlegend=showlegend,
                meanline_visible=True,
                legendgroup=name,
                scalegroup=name
            )
        )
    else:
        fig.add_trace(
            go.Violin(
                y=y,
                name=name,
                box_visible=True,
                showlegend=showlegend,
                meanline_visible=True,
                legendgroup=name,
                scalegroup=name
            )
        )
