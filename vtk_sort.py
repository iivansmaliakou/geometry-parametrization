import numpy as np

def read_vtk_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index where the point data starts
    point_data_index = lines.index('DATASET POLYDATA\n') + 2  # Skipping headers

    # Read the number of points
    num_points = int(lines[point_data_index - 1].split()[1])
    
    # Extract points
    points = []
    for i in range(num_points):
        x, y, z = map(float, lines[point_data_index + i].split())
        points.append((x, y, z))

    return points, lines[:point_data_index-1]

def sort_points_clockwise(points):
    # Project points onto the XY plane
    projected_points = [(x, y) for (x, y, z) in points]

    # Compute the centroid (or use any fixed reference point)
    centroid = np.mean(projected_points, axis=0)
    
    def angle_from_centroid(point):
        x, y = point
        centroid_x, centroid_y = centroid
        return np.arctan2(y - centroid_y, x - centroid_x)
    
    # Sort points based on the angle in descending order (for clockwise)
    sorted_points = sorted(projected_points, key=angle_from_centroid, reverse=True)
    
    return [(x, y, 0) for (x, y) in sorted_points]

def write_vtk_file(file_path, header_lines, sorted_points):
    with open(file_path, 'w') as file:
        file.writelines(header_lines)
        file.write(f'POINTS {len(sorted_points)} double\n')
        for point in sorted_points:
            file.write(f'{point[0]} {point[1]} {point[2]}\n')
        # Write the rest of the data (if needed)
        # For this example, we're just rewriting the points

def vtk_sort_clockwise(filename):
    points, header_lines = read_vtk_file(filename)
    sorted_points = sort_points_clockwise(points)
    write_vtk_file(filename, header_lines, sorted_points)