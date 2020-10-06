import numpy as np
from utils import convertTools as ct


if __name__ == '__main__':
    # upper-triangular binary matrix which used to indicate link information of DAG
    m = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]])

    # vertext type list
    vertexList = ['input','conv3x3-bn-relu','output']

    # use the cell information to construct entire CNN and translate it to ArcText
    arcText = ct.construct_entire_CNN(m, vertexList)

    print('ArcText result:\n%s' %arcText)




