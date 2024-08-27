import vtk


def visualize_vtk_with_vertex_numbers(file_path):
    # Read the .vtk file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the data
    poly_data = reader.GetOutput()

    # Mapper to map the poly data
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    # Actor to represent the poly data
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Renderer
    renderer = vtk.vtkRenderer()

    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the renderer
    renderer.AddActor(actor)

    # Add vertex labels
    point_ids = vtk.vtkIdFilter()
    point_ids.SetInputData(poly_data)
    point_ids.PointIdsOn()
    point_ids.CellIdsOff()
    point_ids.FieldDataOn()
    point_ids.Update()

    label_mapper = vtk.vtkLabeledDataMapper()
    label_mapper.SetInputData(point_ids.GetOutput())
    label_mapper.SetLabelModeToLabelIds()

    label_actor = vtk.vtkActor2D()
    label_actor.SetMapper(label_mapper)
    renderer.AddActor(label_actor)

    # Set background color and size
    renderer.SetBackground(0.1, 0.2, 0.3)  # Background color (RGB)
    render_window.SetSize(800, 600)

    # Start the interactor
    render_window.Render()
    render_window_interactor.Start()


if __name__ == '__main__':
    # Replace 'your_file.vtk' with your actual .vtk file path
    visualize_vtk_with_vertex_numbers('/Users/ivansmaliakou/AGH/praca_magisterska/ROM-with-geometry-informed-snapshots/Example1-Stenosis/data/raw/sample_0/boundary.vtk')