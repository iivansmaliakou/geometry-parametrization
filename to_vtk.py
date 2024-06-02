import numpy as np

pred = np.load('/Users/ivansmaliakou/AGH/praca_magisterska/ROM-with-geometry-informed-snapshots/generated_data/pred.npy')
pred_file = open("/Users/ivansmaliakou/AGH/praca_magisterska/ROM-with-geometry-informed-snapshots/generated_data/pred.vtk", "w")
pred_file.write('# vtk DataFile Version 2.0\n')
pred_file.write('rawdata, Created by Gmsh\n')
pred_file.write('ASCII\n')
pred_file.write('DATASET POLYDATA\n')
pred_file.write('POINTS '+str(pred.shape[0])+' double\n')
for row in pred:
    pred_file.write(str(row[0])+'\t'+str(row[1])+'\t'+str(0)+'\n')


test_sample = np.load('/Users/ivansmaliakou/AGH/praca_magisterska/ROM-with-geometry-informed-snapshots/generated_data/test_sample.npy')
test_sample_file = open("/Users/ivansmaliakou/AGH/praca_magisterska/ROM-with-geometry-informed-snapshots/generated_data/test_sample.vtk", "w")
test_sample_file.write('# vtk DataFile Version 2.0\n')
test_sample_file.write('rawdata, Created by Gmsh\n')
test_sample_file.write('ASCII\n')
test_sample_file.write('DATASET POLYDATA\n')
test_sample_file.write('POINTS '+str(test_sample.shape[0])+' double\n')
for row in test_sample:
    test_sample_file.write(str(row[0])+'\t'+str(row[1])+'\t'+str(0)+'\n')