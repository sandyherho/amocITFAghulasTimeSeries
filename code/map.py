#!/usr/bin/env python
'''
map.py
PyGMT Mapping script
Sandy H. S. Herho <sandy.herho@email.ucr.edu>
2025/05/20
'''
import pygmt

if __name__ == "__main__":
    # Load Earth relief data at 6 arc-minute resolution
    grid = pygmt.datasets.load_earth_relief(resolution='06m')

    # Create a new figure
    fig = pygmt.Figure()

    # Plot the relief data with Robinson projection and geo colormap
    fig.grdimage(grid=grid, projection="R12c", cmap="geo")

    # Add colorbar with elevation labels
    fig.colorbar(frame=["a2500", "x+lElevation", "y+lm"])

    # Define coordinates for three points
    lon = [-68.8, 115.0, 29.0]
    lat = [26.5, -7.5, -36.0]

    # Plot the points as red circles
    fig.plot(x=lon, y=lat, style="c0.2c", fill="red")

    # Display the figure
    fig.show()

    # Save the figure as PNG with high resolution
    fig.savefig('../figs/map.png', dpi=350)
