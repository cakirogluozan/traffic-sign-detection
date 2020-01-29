import numpy as np

def standard_deviation(center, contour, round=True):
    """
    Standard deviation of contour.
    
    Parameters
    ==========
    center : tuple
    contour: list

    Returns
    =======
    distance_std: float
    """

    center_x, center_y = center
    distance_list = list()

    for point in contour:
        x = point[0][0]
        y = point[0][1]
        dist = get_distance((x, y), (center_x, center_y))
        distance_list.append(dist)

    distance_std = np.std(np.array(distance_list))

    if round:
        distance_std = np.round(distance_std, 2)
    return distance_std

def get_distance(point, center):
    """
    Distance between center of contour and a point in contour.

    Parameters
    ==========
    point : tuple
    center: tuple

    Returns
    =======
    distance: float
    """

    x1, y1 = point
    x2, y2 = center
    distance = ((y2-y1)**2 + (x2-x1)**2)**(1/2)

    return distance
