
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon, Rectangle, Wedge
from matplotlib.path import Path
import matplotlib.patches as patches


FONT_SIZE = 12

#1. Draw a sector with/without angle label
def draw_angle_sector(ax, A, B, C, label=None, facecolor='cyan', alpha=0.6):
    ''' A must be the center; B to C must be in counter-clockwise order. Draw the angle sector. '''
    radius = np.sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)
    
    start_angle = np.arctan2(B[1]-A[1], B[0]-A[0])
    end_angle = np.arctan2(C[1]-A[1], C[0]-A[0])
    
    if end_angle < start_angle:
        end_angle += 2*np.pi
    
    start_degree = np.degrees(start_angle)
    end_degree = np.degrees(end_angle)
    
    arc = Arc(A, 2*radius, 2*radius, 
             theta1=start_degree, 
             theta2=end_degree)
    ax.add_patch(arc)

    wedge = Wedge(A, radius, start_degree, end_degree, 
                  facecolor=facecolor, alpha=alpha)
    ax.add_patch(wedge)
    
    # # Draw the two radius lines
    # ax.plot([A[0], B[0]], [A[1], B[1]], 'k-')
    # ax.plot([A[0], C[0]], [A[1], C[1]], 'k-')

    # Annotate the three points
    ax.plot(A[0], A[1], 'ro', label='A')
    ax.plot(B[0], B[1], 'go', label='B')
    ax.plot(C[0], C[1], 'bo', label='C')

    # Annotate the angle
    if label:
        label_x = np.cos(np.radians(start_degree + (end_degree - start_degree) / 2)) * radius * 0.2 + A[0]
        label_y = np.sin(np.radians(start_degree + (end_degree - start_degree) / 2)) * radius * 0.2 + A[1]
        ax.text(label_x, label_y, label, fontsize=FONT_SIZE)


#2. Draw a right angle marker
def draw_right_angle_marker(ax, point1, point2, point3, size_ratio=0.1):
    '''Draw the right angle, point2 is the right angle vertex'''
    vec1 = np.array(point1) - np.array(point2)
    vec2 = np.array(point3) - np.array(point2)

    MAX_X = max([point1[0], point2[0], point3[0]])
    MIN_X = min([point1[0], point2[0], point3[0]])
    box_size = (MAX_X - MIN_X) * size_ratio
    
    # Normalize vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Calculate right angle points
    right_angle_points = [
        np.array(point2),
        np.array(point2) + box_size * vec1,
        np.array(point2) + box_size * (vec1 + vec2),
        np.array(point2) + box_size * vec2,
        np.array(point2)
    ]
    
    right_angle = Polygon(right_angle_points, facecolor='gray', alpha=0.8)
    ax.add_patch(right_angle)


#3. Draw angle marker
def draw_angle_marker(ax, A, B, C, label, facecolor='gray', alpha=0.6, size_ratio=0.1):
    '''Draw an angle marker. A is the center; B to C is counter-clockwise.'''
    B_ = (np.array(B) - np.array(A)) * size_ratio + np.array(A) 
    C_ = (np.array(C) - np.array(A)) * size_ratio + np.array(A) 
    draw_angle_sector(ax, A, B_, C_, label=label, facecolor=facecolor, alpha=alpha)

    
#4. Draw a line
def draw_lines(ax, point_1, point_2):
    '''Draw the line'''
    ax.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], 'k-')


#5. Draw a polygon
def draw_polygon(ax, points, facecolor='gray', alpha=0.6):
    '''Draw the polygon. Points must be in counter-clockwise or clockwise order, all edges must be straight lines.'''
    polygon_points = np.array(points)
    polygon = Polygon(polygon_points, facecolor=facecolor, alpha=alpha)
    ax.add_patch(polygon)
