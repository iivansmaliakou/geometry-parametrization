import numpy as np

pred = np.load('./generated_data/pred_conv.npy')
pred = pred.reshape((360, 2))
pred_file = open("./generated_data/pred_conv.vtk", "w")
pred_file.write('# vtk DataFile Version 2.0\n')
pred_file.write('rawdata, Created by Gmsh\n')
pred_file.write('ASCII\n')
pred_file.write('DATASET POLYDATA\n')
pred_file.write('POINTS '+str(pred.shape[0])+' double\n')
for row in pred:
    pred_file.write(str(row[0])+'\t'+str(row[1])+'\t'+str(0)+'\n')
pred_file.close()


test_sample = np.load('./generated_data/test_sample_conv.npy')
test_sample = test_sample.reshape(360, 2)
test_sample_file = open("./generated_data/test_sample_conv.vtk", "w")
test_sample_file.write('# vtk DataFile Version 2.0\n')
test_sample_file.write('rawdata, Created by Gmsh\n')
test_sample_file.write('ASCII\n')
test_sample_file.write('DATASET POLYDATA\n')
test_sample_file.write('POINTS '+str(test_sample.shape[0])+' double\n')
for row in test_sample:
    test_sample_file.write(str(row[0])+'\t'+str(row[1])+'\t'+str(0)+'\n')
test_sample_file.close()