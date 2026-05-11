TIKZ_CONVERT_EXAMPLES = r"""
# ---- Input ---- #
```latex
\coordinate (A) at (0, 0);
\coordinate (B) at (0, 23.96126);
\coordinate (C) at (22.02582, 23.96126);
\coordinate (D) at (22.02582, 0.0);
\coordinate (E) at (11.01291, -19.07492);
\draw [fill=brown,opacity=0.6](A)--(B)--(C)--(D)--(E)--cycle;
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](A,B,C);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](B,C,D);
\coordinate (F) at (22.02582, 41.27738);
\coordinate (G) at (0.0, 41.27738);
\coordinate (H) at (14.9962, 32.61932);
\draw [fill=cyan,opacity=0.6](B)--(C)--(F)--(G)--(H)--cycle;
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](F,C,B);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](G,F,C);
\coordinate (I) at (37.02202, 32.61932);
\tkzMarkAngle[fill=gray,size=1.0,opacity=.8](I,C,F);
\tkzLabelAngle[pos=1.65,font=\Huge](I,C,F){$60^\circ$};
\pgfmathanglebetweenpoints{\pgfpointanchor{C}{center}}{\pgfpointanchor{I}{center}}
\let\StartAngle\pgfmathresult

\pgfmathanglebetweenpoints{\pgfpointanchor{C}{center}}{\pgfpointanchor{F}{center}}
\let\EndAngle\pgfmathresult

\pgfmathsetmacro{\EndAngleUpd}{ifthenelse(\StartAngle > \EndAngle,360+\EndAngle,\EndAngle))}
\draw[fill=purple,opacity=0.6] (C) -- (I)arc [start angle=\StartAngle, end angle=\EndAngleUpd, radius=17.31611713939158] -- (C);
\draw (0, 0) -- node[left,xshift=-5mm,pos=3.2325,font=\Huge](){A}(0, 0);
\draw (0, 23.96126) -- node[left,xshift=-5mm,pos=5.52904,font=\Huge](){B}(0, 23.96126);
\draw (22.02582, 23.96126) -- node[right,xshift=5mm,pos=5.52904,font=\Huge](){C}(22.02582, 23.96126);
\draw (22.02582, 0.0) -- node[right,xshift=5mm,pos=3.2325,font=\Huge](){D}(22.02582, 0.0);
\draw (11.01291, -19.07492) -- node[below,yshift=-5mm,pos=6.45956,font=\Huge](){E}(11.01291, -19.07492);
\draw (22.02582, 41.27738) -- node[above,yshift=5mm,pos=9.66589,font=\Huge](){F}(22.02582, 41.27738);
\draw (0.0, 41.27738) -- node[above,yshift=5mm,pos=9.66589,font=\Huge](){G}(0.0, 41.27738);
\draw (14.9962, 32.61932) -- node[above,yshift=5mm,pos=7.05735,font=\Huge](){H}(14.9962, 32.61932);
\draw (37.02202, 32.61932) -- node[right,xshift=5mm,pos=9.71922,font=\Huge](){I}(37.02202, 32.61932);
```
# ---- Output ---- #
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 12))

# Coordinates
A = (0, 0)
B = (0, 23.96126)
C = (22.02582, 23.96126)
D = (22.02582, 0)
E = (11.01291, -19.07492)
F = (22.02582, 41.27738)
G = (0, 41.27738)
H = (14.9962, 32.61932)
I = (37.02202, 32.61932)

# Draw the brown polygon
brown_polygon = patches.Polygon(
    [A, B, C, D, E],
    closed=True,
    facecolor="brown",
    edgecolor="black",
    alpha=0.6
)
ax.add_patch(brown_polygon)

# Draw the cyan polygon
cyan_polygon = patches.Polygon(
    [B, C, F, G, H],
    closed=True,
    facecolor="cyan",
    edgecolor="black",
    alpha=0.6
)
ax.add_patch(cyan_polygon)

# Draw the purple arc
radius = 17.31612  # Corrected radius from the TikZ code
start_angle = np.degrees(np.arctan2(I[1] - C[1], I[0] - C[0]))
end_angle = np.degrees(np.arctan2(F[1] - C[1], F[0] - C[0]))

if start_angle > end_angle:
    end_angle += 360

arc = patches.Wedge(
    center=C,
    r=radius,
    theta1=start_angle,
    theta2=end_angle,
    facecolor="purple",
    edgecolor="black",
    alpha=0.6
)
ax.add_patch(arc)

# Draw right-angle markers
def draw_right_angle(ax, p1, p2, p3, size=1, color="gray", alpha=0.8):
    '''Draws a right-angle marker at the corner defined by p1, p2, and p3.'''
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v1 = v1 / np.linalg.norm(v1) * size
    v2 = v2 / np.linalg.norm(v2) * size
    marker = patches.Polygon(
        [p2, p2 + v1, p2 + v1 + v2, p2 + v2],
        closed=True,
        facecolor=color,
        edgecolor=None,
        alpha=alpha
    )
    ax.add_patch(marker)

# Draw the 60-degree angle marker
def draw_angle_marker(ax, center, p1, p2, radius, angle_text, color="gray", alpha=0.8):
    '''Draws an angle marker and label.'''
    angle1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    angle2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
    
    if angle2 < angle1:
        angle2 += 2 * np.pi

    theta = np.linspace(angle1, angle2, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.fill(x, y, color=color, alpha=alpha)

    # Add angle text
    mid_angle = (angle1 + angle2) / 2
    text_x = center[0] + (radius + 1) * np.cos(mid_angle)
    text_y = center[1] + (radius + 1) * np.sin(mid_angle)
    ax.text(text_x, text_y, angle_text, fontsize=14, ha="center", va="center")

draw_angle_marker(ax, C, I, F, radius=2, angle_text=r"$60^\circ$")

# Add labels
labels = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "H": H, "I": I}
for label, coord in labels.items():
    ax.text(coord[0], coord[1] + 1, label, fontsize=12, ha="center", va="center")

# -- RIGHT ANGLE ANNOTATION -- #
draw_right_angle(ax, A, B, C)
draw_right_angle(ax, B, C, D)
draw_right_angle(ax, F, C, B)
draw_right_angle(ax, G, F, C)

# Adjust axis
ax.set_xlim(-5, 45)  # Adjusted x-limits for better presentation
ax.set_ylim(-25, 50)  # Adjusted y-limits for better presentation
ax.set_aspect('equal', adjustable='datalim')
ax.axis('off')

# Show plot
plt.show()
```
# ---- Input ---- #
```latex
\coordinate (A) at (0, 0);
\coordinate (B) at (0, 20);
\coordinate (C) at (13.57486, 0.0);
\draw [fill=orange,opacity=0.6](A)--(B)--(C)--cycle;
\draw (0.0, 10.0) -- node[left,xshift=-5mm,pos=1.46124,font=\Huge](){20}(0.0, 10.0);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](C,A,B);
\coordinate (D) at (13.57486, -12.49);
\coordinate (E) at (0.0, -12.49);
\coordinate (F) at (10.81665, -6.245);
\draw [fill=yellow,opacity=0.6](A)--(C)--(D)--(E)--(F)--cycle;
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](A,C,D);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](C,D,E);
\coordinate (G) at (23.57486, 0.0);
\coordinate (H) at (23.57486, -12.49);
\draw [fill=lime,opacity=0.6](C)--(G)--(H)--(D)--cycle;
\draw (18.57486, 0.0) -- node[above,yshift=5mm,pos=1.6237,font=\Huge](){10}(18.57486, 0.0);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](C,G,H);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](G,H,D);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](H,D,C);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](D,C,G);
\draw (0, 0) -- node[left,xshift=-5mm,pos=2.09489,font=\Huge](){A}(0, 0);
\draw (0, 20) -- node[above,yshift=5mm,pos=3.66086,font=\Huge](){B}(0, 20);
\draw (13.57486, 0.0) -- node[right,xshift=5mm,pos=2.92249,font=\Huge](){C}(13.57486, 0.0);
\draw (13.57486, -12.49) -- node[below,yshift=-5mm,pos=5.50856,font=\Huge](){D}(13.57486, -12.49);
\draw (0.0, -12.49) -- node[below,yshift=-5mm,pos=5.11779,font=\Huge](){E}(0.0, -12.49);
\draw (10.81665, -6.245) -- node[below,yshift=-5mm,pos=3.73439,font=\Huge](){F}(10.81665, -6.245);
\draw (23.57486, 0.0) -- node[right,xshift=5mm,pos=5.24752,font=\Huge](){G}(23.57486, 0.0);
\draw (23.57486, -12.49) -- node[right,xshift=5mm,pos=7.02422,font=\Huge](){H}(23.57486, -12.49);
```
# ---- Output ---- #
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Coordinates
A = (0, 0)
B = (0, 20)
C = (13.57486, 0)
D = (13.57486, -12.49)
E = (0, -12.49)
F = (10.81665, -6.245)
G = (23.57486, 0)
H = (23.57486, -12.49)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))

# Adjust axis limits
ax.set_xlim(-5, 30)
ax.set_ylim(-20, 25)

# Draw the orange triangle (A, B, C)
orange_triangle = patches.Polygon([A, B, C], closed=True, color='orange', alpha=0.6)
ax.add_patch(orange_triangle)

# Draw the yellow polygon (A, C, D, E, F)
yellow_polygon = patches.Polygon([A, C, D, E, F], closed=True, color='yellow', alpha=0.6)
ax.add_patch(yellow_polygon)

# Draw the lime rectangle (C, G, H, D)
lime_polygon = patches.Polygon([C, G, H, D], closed=True, color='lime', alpha=0.6)
ax.add_patch(lime_polygon)

# Draw right-angle markers
def draw_right_angle(ax, p1, p2, p3, size=1, color="gray", alpha=0.8):
    '''Draws a right-angle marker at the corner defined by p1, p2, and p3.'''
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v1 = v1 / np.linalg.norm(v1) * size
    v2 = v2 / np.linalg.norm(v2) * size
    marker = patches.Polygon(
        [p2, p2 + v1, p2 + v1 + v2, p2 + v2],
        closed=True,
        facecolor=color,
        edgecolor=None,
        alpha=alpha
    )
    ax.add_patch(marker)

# Add labels
ax.text(A[0], A[1], 'A', fontsize=16, ha='right', va='top')
ax.text(B[0], B[1], 'B', fontsize=16, ha='right', va='bottom')
ax.text(C[0], C[1], 'C', fontsize=16, ha='left', va='top')
ax.text(D[0], D[1], 'D', fontsize=16, ha='left', va='top')
ax.text(E[0], E[1], 'E', fontsize=16, ha='right', va='top')
ax.text(F[0], F[1], 'F', fontsize=16, ha='center', va='top')
ax.text(G[0], G[1], 'G', fontsize=16, ha='left', va='top')
ax.text(H[0], H[1], 'H', fontsize=16, ha='left', va='top')

# -- RIGHT ANGLE ANNOTATION -- #
draw_right_angle(ax, C, A, B)
draw_right_angle(ax, A, C, D)
draw_right_angle(ax, C, D, E)
draw_right_angle(ax, C, G, H)
draw_right_angle(ax, G, H, D)
draw_right_angle(ax, H, D, C)
draw_right_angle(ax, D, C, G)

# -- LENGTH ANNOTATION -- #
ax.text((A[0] + B[0]) / 2, (A[1] + B[1]) / 2, '20', fontsize=16, ha='right', va='center')
ax.text((C[0] + G[0]) / 2, (C[1] + G[1]) / 2, '10', fontsize=16, ha='center', va='bottom')

# Configure grid and aspect ratio
ax.grid(False)
ax.set_aspect('equal')
ax.axis('off')

plt.show()
```
# ---- Input ---- #
```latex
\coordinate (A) at (0, 0);
\coordinate (B) at (0, 12);
\coordinate (C) at (6.20799, 6.5162);
\draw [fill=green,opacity=0.6](A)--(B)--(C)--cycle;
\draw (3.10399, 3.2581) -- node[below,yshift=-5mm,pos=0.80397,font=\Huge](){9}(3.10399, 3.2581);
\draw (0.0, 6.0) -- node[left,xshift=-5mm,pos=0.7,font=\Huge](){12}(0.0, 6.0);
\coordinate (D) at (0.3197, 27.71356);
\draw [fill=blue,opacity=0.6](B)--(C)--(D)--cycle;
\draw (3.26385, 17.11488) -- node[right,xshift=5mm,pos=0.7,font=\Huge](){22}(3.26385, 17.11488);
\coordinate (E) at (-7.67864, 27.8763);
\coordinate (F) at (-7.99834, 12.16273);
\draw [fill=yellow,opacity=0.6](B)--(D)--(E)--(F);
\draw (-3.67947, 27.79493) -- node[above,yshift=5mm,pos=2.24382,font=\Huge](){8}(-3.67947, 27.79493);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](E,D,B);
\tkzMarkRightAngle[fill=gray,size=0.5,opacity=.8](F,E,D);
\draw [white,opacity=1.0](B)--(F);
\coordinate (G) at ($(F)!.5!(B)$);
\pgfmathanglebetweenpoints{\pgfpointanchor{G}{center}}{\pgfpointanchor{B}{center}}
\let\StartAngle\pgfmathresult

\pgfmathanglebetweenpoints{\pgfpointanchor{G}{center}}{\pgfpointanchor{F}{center}}
\let\EndAngle\pgfmathresult

\pgfmathsetmacro{\EndAngleUpd}{ifthenelse(\StartAngle > \EndAngle,360+\EndAngle,\EndAngle))}
\draw[fill=white,opacity=1.0] (B) arc [start angle=\StartAngle, end angle=\EndAngleUpd, radius=4.0];
\draw (0, 0) -- node[below,yshift=-5mm,pos=1.69253,font=\Huge](){A}(0, 0);
\draw (0, 12) -- node[above,yshift=5mm,pos=1.60795,font=\Huge](){B}(0, 12);
\draw (6.20799, 6.5162) -- node[right,xshift=5mm,pos=1.07976,font=\Huge](){C}(6.20799, 6.5162);
\draw (0.3197, 27.71356) -- node[above,yshift=5mm,pos=5.61923,font=\Huge](){D}(0.3197, 27.71356);
\draw (-7.67864, 27.8763) -- node[above,yshift=5mm,pos=6.18612,font=\Huge](){E}(-7.67864, 27.8763);
\draw (-7.99834, 12.16273) -- node[left,xshift=-5mm,pos=3.04595,font=\Huge](){F}(-7.99834, 12.16273);
```
# ---- Output ---- #
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

# Coordinates
A = (0, 0)
B = (0, 12)
C = (6.20799, 6.5162)
D = (0.3197, 27.71356)
E = (-7.67864, 27.8763)
F = (-7.99834, 12.16273)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 14))

# Adjust axis limits for better presentation
ax.set_xlim(-15, 15)
ax.set_ylim(-5, 35)

# Draw the green triangle (A, B, C)
green_triangle = patches.Polygon([A, B, C], closed=True, color="green", alpha=0.6)
ax.add_patch(green_triangle)

# Draw the blue triangle (B, C, D)
blue_triangle = patches.Polygon([B, C, D], closed=True, color="blue", alpha=0.6)
ax.add_patch(blue_triangle)

# Draw the yellow clipped area
arc_center = ((B[0] + F[0]) / 2, (B[1] + F[1]) / 2)  # Midpoint of B and F
arc_radius = np.sqrt((B[0] - F[0])**2 + (B[1] - F[1])**2) / 2
theta = np.linspace(0, np.pi, 100)  # Arc angles (for a semicircle)

# Arc boundary points
arc_points = np.array([
    (arc_center[0] + arc_radius * np.cos(t), arc_center[1] + arc_radius * np.sin(t))
    for t in theta
])

# Combine arc points with parts of the polygon to clip
polygon_points = np.vstack([B, D, E, F])  # Yellow quadrilateral points
clipped_points = np.vstack([arc_points, np.flip(polygon_points, axis=0)])

# Create a Path for the clipped yellow area
clipped_path = Path(clipped_points, closed=True)
clipped_patch = patches.PathPatch(clipped_path, facecolor="yellow", alpha=0.6)
ax.add_patch(clipped_patch)

# Draw right-angle markers
def draw_right_angle(ax, p1, p2, p3, size=1.0, color="gray", alpha=0.8):
    '''Draw a right-angle marker at the corner defined by p1, p2, and p3.'''
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v1 = v1 / np.linalg.norm(v1) * size
    v2 = v2 / np.linalg.norm(v2) * size
    marker = patches.Polygon(
        [p2, p2 + v1, p2 + v1 + v2, p2 + v2],
        closed=True,
        facecolor=color,
        edgecolor=None,
        alpha=alpha
    )
    ax.add_patch(marker)

# Add right-angle markers
draw_right_angle(ax, E, D, B)  # Right angle at D
draw_right_angle(ax, F, E, D)  # Right angle at E

# Add labels
ax.text(A[0], A[1], "A", fontsize=16, ha="right", va="top")
ax.text(B[0], B[1], "B", fontsize=16, ha="right", va="bottom")
ax.text(C[0], C[1], "C", fontsize=16, ha="left", va="top")
ax.text(D[0], D[1], "D", fontsize=16, ha="left", va="bottom")
ax.text(E[0], E[1], "E", fontsize=16, ha="right", va="top")
ax.text(F[0], F[1], "F", fontsize=16, ha="left", va="top")

# -- LENGTH ANNOTATION -- #
ax.text((A[0] + C[0]) / 2, (A[1] + C[1]) / 2, "9", fontsize=16, ha="center", va="top")
ax.text((A[0] + B[0]) / 2, (A[1] + B[1]) / 2, "12", fontsize=16, ha="right", va="center")  # Label for side BC
ax.text((C[0] + D[0]) / 2, (C[1] + D[1]) / 2, "22", fontsize=16, ha="left", va="center")  # Label for side BD
ax.text((D[0] + E[0]) / 2, (D[1] + E[1]) / 2, "8", fontsize=16, ha="center", va="bottom")  # Label for side DE

# Configure the plot
ax.set_aspect("equal")
ax.axis("off")

# Show the plot
plt.show()
```
"""