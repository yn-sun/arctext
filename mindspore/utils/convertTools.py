import copy

import numpy as np

from utils import graph, units, api

import pandas as pd

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'

def compute_vertex_channels(input_channel, output_channel, matrix):
    """
    This function is used to compute the input channel and output channel of layers in a cell

    :param input_channel: the input channel count of the cell
    :param output_channel: the output channel count of the cell
    :param matrix: adjacency matrix of the vertices in cell
    :return: list of input channel counts and output channel counts, in order of the vertices
    """
    # dimension of the matrix
    num_vertices = np.shape(matrix)[0]
    # initialize input and output channels
    vertex_out_channels = [0] * num_vertices
    vertex_in_channels = [0] * num_vertices

    # initialize the input and output channel at input and output vertex
    # The in_channels of other vertices are determined by the out_channel of the previous vertex. So we set in_channel as 3 for stem layer(the first layer of the CNN)
    vertex_in_channels[0] = 3
    vertex_in_channels[num_vertices - 1] = output_channel
    vertex_out_channels[0] = input_channel
    vertex_out_channels[num_vertices - 1] = output_channel

    # module only has input and output vertices
    if num_vertices == 2:
        return vertex_in_channels, vertex_out_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channel // in_degree[num_vertices - 1]
    correction = output_channel % in_degree[num_vertices - 1]

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_out_channels[v] = interior_channels
            if correction:
                vertex_out_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_out_channels[v] = max(vertex_out_channels[v], vertex_out_channels[dst])
        assert vertex_out_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_out_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_out_channels[v] >= vertex_out_channels[dst]
    assert final_fan_in == output_channel or num_vertices == 2

    # set all in_channels for vertices according to the out_channels of previous vertices
    for v in range(0, num_vertices - 1):
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                vertex_in_channels[dst] = vertex_out_channels[v]

    return vertex_in_channels, vertex_out_channels

def convert_matrix_to_graph(matrix, vertexList, inputUnit, vertexIndex, inputChannel, outputChannel, width_heightStr):
    """
    Given the information of cell(matrix, type list of vertices, etc.), this function convert it into a Graph type.

    :param matrix: adjacent matrix of cell
    :param vertexList: type list of vertices
    :param inputUnit: input unit of cell, which is convenient for constructing the entire CNN.
    :param vertexIndex: the start vertex index for the first vertex in the graph
    :param inputChannel: input channel count of the cell
    :param outputChannel: output channel count of the cell
    :param width_heightStr: width and height of tensor, which is used to construct the 'in_size' and 'out_size' information of arcText unit
    :return: Graph object that represents a cell
    """

    # calculate input channel and output channel counts for vertices
    inchannels, outchannels = compute_vertex_channels(inputChannel, outputChannel, matrix)

    # initialize a Graph object
    G = graph.Graph()
    # record available id for new vertex
    idIndicator = len(vertexList)
    # there may be several arcText unit for a single layer(e.g. conv1X1-BN-ReLU has 3 arcText unit)
    # this dictionary records the connect information in each layer, which is used to construct connection relationship in the Graph
    connect_dict = {}

    # convert each vertex according to its type
    for v in range(0, len(vertexList)):
        # if there is no extra information in the layer, the connect_dict will add itself to the dict
        connect_dict[v] = v
        # if vertex is INPUT, the inputUnit will be added into the Graph
        if vertexList[v] == INPUT:
            unit = inputUnit
            inputUnit.key = v
            G.addVertex(unit)

        # if vertex is OUTPUT
        if vertexList[v] == OUTPUT:
            # creat a conv_arcunit to represent the output vertex and assign proper attributes for the unit.
            unit = units.conv_arcunit(v)
            unit.in_size = width_heightStr + str(inchannels[v])
            unit.out_size = width_heightStr + str(outchannels[v])
            unit.kernel = '1-1'
            unit.padding = '0-0-0-0-0-0-0-0'

            # a convolution layer in the NASBench has BN and ReLU, create these two Unit for the vertex
            unitBN = units.mf_arcunit(idIndicator)
            idIndicator = idIndicator + 1
            unitBN.name = 'BN'
            unitBN.in_size = width_heightStr + str(inchannels[v])
            unitBN.out_size = width_heightStr + str(outchannels[v])

            unitReLU = units.mf_arcunit(idIndicator)
            #  the ReLU unit is the "output" of this layer.
            #  E.g. Supposed layer A has 3 unit A.conv1, A.BN, A.ReLU and A connect to B in the Graph. This dict will
            #  record A.ReLU as the output of layer A, and use this information to construct connection to B in the folloing.
            connect_dict[v] = idIndicator
            idIndicator = idIndicator + 1 # update the idIndicator
            unitReLU.name = 'ReLU'
            unitReLU.in_size = width_heightStr + str(inchannels[v])
            unitReLU.out_size = width_heightStr + str(outchannels[v])
            # add the vertex and their relations to the Graph object
            G.addEdge(unit, unitBN)
            G.addEdge(unitBN, unitReLU)

        # if the vertex is CONV1X1
        if vertexList[v] == CONV1X1:
            unit = units.conv_arcunit(v)
            unit.in_size = width_heightStr + str(inchannels[v])
            unit.out_size = width_heightStr + str(outchannels[v])
            unit.kernel = '1-1'
            unit.padding = '0-0-0-0-0-0-0-0'

            unitBN = units.mf_arcunit(idIndicator)
            idIndicator = idIndicator + 1
            unitBN.name = 'BN'
            unitBN.in_size = width_heightStr + str(inchannels[v])
            unitBN.out_size = width_heightStr + str(outchannels[v])

            unitReLU = units.mf_arcunit(idIndicator)
            connect_dict[v] = idIndicator
            idIndicator = idIndicator + 1
            unitReLU.name = 'ReLU'
            unitReLU.in_size = width_heightStr + str(inchannels[v])
            unitReLU.out_size = width_heightStr + str(outchannels[v])

            G.addEdge(unit, unitBN)
            G.addEdge(unitBN, unitReLU)

        # if the vertex is CONV3X3
        if vertexList[v] == CONV3X3:
            unit = units.conv_arcunit(v)
            unit.in_size = width_heightStr + str(inchannels[v])
            unit.out_size = width_heightStr + str(outchannels[v])
            unit.kernel = '3-3'
            unit.padding = '1-0-1-0-1-0-1-0'

            unitBN = units.mf_arcunit(idIndicator)
            idIndicator = idIndicator + 1
            unitBN.name = 'BN'
            unitBN.in_size = width_heightStr + str(inchannels[v])
            unitBN.out_size = width_heightStr + str(outchannels[v])

            unitReLU = units.mf_arcunit(idIndicator)
            connect_dict[v] = idIndicator
            idIndicator = idIndicator + 1
            unitReLU.name = 'ReLU'
            unitReLU.in_size = width_heightStr + str(inchannels[v])
            unitReLU.out_size = width_heightStr + str(outchannels[v])
            G.addEdge(unit, unitBN)
            G.addEdge(unitBN, unitReLU)

        # if the vertex is MAXPOOL3X3
        if vertexList[v] == MAXPOOL3X3:
            unit = units.pool_arcunit(v)
            unit.in_size = width_heightStr + str(inchannels[v])
            unit.out_size = width_heightStr + str(outchannels[v])
            unit.padding = '1-1-1-1'
            G.addVertex(unit)

    # construct the relations according to the matrix and connect_dict
    for f in range(0, len(vertexList)):
        for t in range(f, len(vertexList)):
            if matrix[f, t]:
                if not G.getVertex(t):
                    print(t)
                G.addEdge(G.getVertex(connect_dict[f]), G.getVertex(t))

    # assign 'id' and 'connect_to' information for units
    G.assign_id_for_graph(vertexIndex)

    return G

def construct_entire_CNN(matrix, vertexList):
    """
    This function  use adjacent matrix and type list of vertices of the cell to construct an entire CNN
    If you want to learn more details about cells and structure, you can read the paper <NAS-Bench-101: Towards Reproducible Neural Architecture Search>

    :param matrix: adjacent matrix of cell
    :param vertexList: type list of vertices
    :return: an arcText string of the CNN
    """
    # construct the initial layer of the model: stem
    width_heightStr1 = '32-32-'
    stem = units.conv_arcunit(0)
    stem.in_size = width_heightStr1 + '3'
    stem.out_size = width_heightStr1 + '128'
    stem.kernel = '3-3'
    stem.padding = '1-0-1-0-1-0-1-0'
    vertexIndex = 0

    arcText = ''
    # stack 1
    # construct 3 cells in the first stack. When constructing the next cell, we use the vertex in last cell to be the
    # first cell in the next cell. Therefore, there are endunits in the code
    G1 = convert_matrix_to_graph(matrix, vertexList, stem, 0, 128, 128, width_heightStr1)
    vertexIndex = vertexIndex + G1.vSize - 1
    endunit1 = copy.deepcopy(G1.vertexList[G1.vSize - 1])
    G2 = convert_matrix_to_graph(matrix, vertexList, endunit1, vertexIndex, 128, 128, width_heightStr1)
    vertexIndex = vertexIndex + G2.vSize - 1
    endunit2 = copy.deepcopy(G2.vertexList[G2.vSize - 1])
    G3 = convert_matrix_to_graph(matrix, vertexList, endunit2, vertexIndex, 128, 128, width_heightStr1)
    vertexIndex = vertexIndex + G3.vSize
    endunit3 = copy.deepcopy(G3.vertexList[G3.vSize - 1])
    endunit3.connect_to = str(vertexIndex)


    arcText = arcText + G1.do_print() + '\n' + G2.do_print() + '\n' + G3.do_print() + '\n' + str(endunit3) + '\n'


    # downsample structure in CNN
    pool1 = units.pool_arcunit(1)
    pool1.id = vertexIndex
    pool1.padding = '0-0-0-0'
    pool1.stride = '2-2'
    pool1.kernel = '2-2'
    pool1.in_size = '32-32-128'
    pool1.out_size = '16-16-256'

    # stack 2
    width_heightStr1 = '16-16-'
    G2_1 = convert_matrix_to_graph(matrix, vertexList, pool1, vertexIndex, 256, 256, width_heightStr1)
    vertexIndex = vertexIndex + G2_1.vSize - 1
    endunit2_1 = copy.deepcopy(G2_1.vertexList[G2_1.vSize - 1])
    G2_2 = convert_matrix_to_graph(matrix, vertexList, endunit2_1, vertexIndex, 256, 256, width_heightStr1)
    vertexIndex = vertexIndex + G2_2.vSize - 1
    endunit2_2 = copy.deepcopy(G2_2.vertexList[G2_2.vSize - 1])
    G2_3 = convert_matrix_to_graph(matrix, vertexList, endunit2_2, vertexIndex, 256, 256, width_heightStr1)
    vertexIndex = vertexIndex + G2_3.vSize
    endunit2_3 = copy.deepcopy(G2_3.vertexList[G2_3.vSize - 1])
    endunit2_3.connect_to = str(vertexIndex)

    arcText = arcText + G2_1.do_print() + '\n' + G2_2.do_print() + '\n' + G2_3.do_print() + '\n' + str(endunit2_3) + '\n'

    # the second downsample layer
    pool2 = units.pool_arcunit(2)
    pool2.id = vertexIndex
    pool2.padding = '0-0-0-0'
    pool2.stride = '2-2'
    pool2.kernel = '2-2'
    pool2.in_size = '16-16-256'
    pool2.out_size = '8-8-512'

    # stack 3
    width_heightStr1 = '8-8-'
    G3_1 = convert_matrix_to_graph(matrix, vertexList, pool2, vertexIndex, 512, 512, width_heightStr1)
    vertexIndex = vertexIndex + G3_1.vSize - 1
    endunit3_1 = copy.deepcopy(G3_1.vertexList[G3_1.vSize - 1])
    G3_2 = convert_matrix_to_graph(matrix, vertexList, endunit3_1, vertexIndex, 512, 512, width_heightStr1)
    vertexIndex = vertexIndex + G3_2.vSize - 1
    endunit3_2 = copy.deepcopy(G3_2.vertexList[G3_2.vSize - 1])
    G3_3 = convert_matrix_to_graph(matrix, vertexList, endunit3_2, vertexIndex, 512, 512, width_heightStr1)
    vertexIndex = vertexIndex + G3_3.vSize
    endunit3_3 = copy.deepcopy(G3_3.vertexList[G3_3.vSize - 1])
    endunit3_3.connect_to = str(vertexIndex)

    # final average pool layer
    poolavg = units.pool_arcunit(3)
    poolavg.id = vertexIndex
    poolavg.type = 'Avg'
    poolavg.padding = '0-0-0-0'
    poolavg.stride = '1-1'
    poolavg.kernel = '8-8'
    poolavg.in_size = '8-8-512'
    poolavg.out_size = '1-1-512'
    arcText = arcText + G3_1.do_print() + '\n' + G3_2.do_print() + '\n' + G3_3.do_print() + '\n' + str(
        endunit3_3) + '\n' + str(poolavg)


    return arcText

def convert_arcText_to_graph(arcText):
    """
    this function is used to convert an arcText to a graph
    :param arcText: arcText string
    :return: a Graph object
    """
    texts = arcText.split('\n')
    G = graph.Graph()
    for item in texts:
        key = int(item[item.index("id:") + 3:item.index(";")])
        value = item[item.index(";") + 1:item.index("connect")]
        connect_to = item[item.index("connect_to:") + 11: len(item)]
        tempUnit = units.unit(key)
        tempUnit.value = value
        if connect_to != '':
            tempUnit.connect_to_list = [int(i) for i in connect_to.split(',')]
        else:
            tempUnit.connect_to_list = []
        G.addVertex(tempUnit)
    return G

def update_id_for_arcText(arcText):
    """
    this function is used to find proper id for arcText (Algorithm 2 in the paper)
    :param arcText: arcText string
    :return: arcText string with updated id
    """
    G = convert_arcText_to_graph(arcText)
    return G.get_ordered_arcText()

def convert_NASBench_to_arcTextDf(NASBENCH_JSON="path/nasbench_only108.json", record_size=40000, save_path="data/arcText.csv"):
    """
    convert NASBench data set to arcText dataframe
    :param NASBENCH_JSON: the path of 'nasbench_only108.json'
    :param record_size: expected record size
    :param save_path: save path of arcText dataframe
    :return: NaN
    """
    # get nasbench dataset from file
    nasbench = api.NASBench(NASBENCH_JSON)
    # read json file
    record_count = 0
    # initialize list to save attributes
    hashlist = []
    textlist = []
    accuracylist = []
    timelist = []

    for unique_hash in nasbench.hash_iterator():
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        # Get single arcText from matrix and cell type list
        arcText = construct_entire_CNN(fixed_metrics['module_adjacency'], fixed_metrics['module_operations'])
        arcText = update_id_for_arcText(arcText)
        # get train time and accuracy of each CNN
        final_training_time_list = []
        final_test_accuracy_list = []
        for i in range(3):
            final_training_time_list.append(computed_metrics[108][i]['final_training_time'])
            final_test_accuracy_list.append(computed_metrics[108][i]['final_test_accuracy'])
        # use the mean of three metrics
        final_training_time = np.mean(final_training_time_list)
        final_test_accuracy = np.mean(final_test_accuracy_list)

        print('Converting NASBench to ArcText: %d / %d' % (record_count, record_size))
        record_count = record_count + 1
        hashlist.append(unique_hash)
        textlist.append(arcText)
        accuracylist.append(final_test_accuracy)
        timelist.append(final_training_time)

        if len(hashlist) > record_size:
            break

    # save dataframe to the file
    arcTextdf = pd.DataFrame({'hash': hashlist, 'text': textlist, 'accuracy': accuracylist, 'time': timelist})
    arcTextdf.to_csv(save_path)

def get_unique_unit_list(arcText):
    """
    get unique unit list in the arcText, which is used to construct onehot dictionary
    :param arcText: arcText string
    :return: unique unit string list
    """
    texts = arcText.split('\n')
    unique_list = []
    for item in texts:
        start = item.index("in_size")
        end = item.index("connect")
        unique_list.append(item[start:end])

    unique_list = list(set(unique_list))
    return unique_list

def get_onehot_dict_from_arcTextDf(csv_path="data/arcText.csv", save_path='data/onehot_dict.txt'):
    """
    get onehot dictionary from arcText dataframe, which used to convert arcText to onehot vector
    :param csv_path: csv file path of arcText dataframe
    :param save_path: the path to save onehot dictionary
    :return: max number of unit in arcText, which used to fill zeros when encoding onehot
    """
    data_iter = pd.read_csv(csv_path, chunksize=20000)
    unique_list = []
    max_arcText_len = 0
    for data_chunk in data_iter:
        tempdf = data_chunk
        tempdf['len'] = tempdf['text'].apply(lambda x: len(x.split('\n')))
        tempdf['unique_list'] = tempdf['text'].apply(lambda x: get_unique_unit_list(x))
        temp_max_len = max(tempdf['len'])
        unique_lists = list(tempdf['unique_list'])
        if max_arcText_len < temp_max_len:
            max_arcText_len = temp_max_len
        for listitem in unique_lists:
            unique_list.extend(listitem)
            unique_list = list(set(unique_list))
    onehot_dict = {}
    for i in range(0, len(unique_list)):
        onehot_dict[unique_list[i]] = i
    # 保存
    f = open(save_path, 'w')
    f.write(str(onehot_dict))
    f.close()
    return max_arcText_len

def convert_text_to_onehot(arcText, onehot_dict, max_arcText_len):
    """
    this function converts arcText to onehot vector
    :param arcText: arcText string
    :param onehot_dict: onehot dictionary
    :param max_arcText_len: max number of unit in arcText, which used to fill zeros when encoding onehot
    :return onehot vector of arcText
    """
    texts = arcText.split('\n')

    onehot = []
    for item in texts:
        start = item.index("in_size:")
        end = item.index("connect")
        onehot.append(onehot_dict[item[start:end]])
    # fill zeros at the end of vector to ensure all vectors have same length
    onehot.extend([0] * (max_arcText_len - len(onehot)))
    return onehot

def generate_dataset_from_arcTextDf_and_onehoDict(csv_path="data/arcText.csv", dict_path='data/onehot_dict.txt',
                                                  save_path="data/dataset.csv",max_arcText_len = 166):
    """
    generate dataset from arcText dataframe and onehot dictionary, which is used to train and test model
    :param csv_path: csv path of arcText dataframe
    :param dict_path: onehot dictionary path
    :param save_path: dataset path
    :return: NaN
    """
    # read onehot dictionary
    f = open(dict_path, 'r')
    a = f.read()
    onehot_dict = eval(a)
    f.close()
    # read dataframe
    data_iter = pd.read_csv(csv_path, chunksize=20000)
    # initialize lists to save attributes in dataset
    onehot_list = []
    accuracy_list = []


    for data_chunk in data_iter:
        tempdf = data_chunk
        tempdf['onehot'] = tempdf['text'].apply(lambda x: convert_text_to_onehot(x, onehot_dict, max_arcText_len))
        onehot_list.extend(list(tempdf['onehot']))
        accuracy_list.extend(list(tempdf['accuracy']))

    datasetdf = pd.DataFrame({'x': onehot_list, 'y': accuracy_list})

    # need to stop here
    datasetdf = datasetdf.drop(0, axis=0, inplace=False)

    # split vectors to independent columns
    datasetdf['x'] = datasetdf['x'].apply(lambda x: str(x).replace("[", "").replace("]", ""))
    datasetdf = pd.concat([datasetdf, datasetdf['x'].str.split(',', expand=True)], axis=1)
    datasetdf.to_csv(save_path)

