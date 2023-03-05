import hashlib


class Graph():
    """
    Graph object, which is used to represent arcText units and their relationships
    """

    def __init__(self):
        self.vertexList = {}  # save vertices in the graph(units in the arcText)
        self.vSize = 0  # vertex count
        self.indegreeDict = {}  # save the in-degree of vertices in the graph
        self.topoDict = {}  # save topological order in the graph

    def addVertex(self, unit):
        """
        add a vertex(unit) in the graph
        :param unit: the unit which will be added into graph
        :return: Nan
        """
        self.vertexList.update({unit.key: unit})
        self.vSize += 1

    def addEdge(self, start_vertex, end_vertex):
        """
        Add an edge to the graph. If vertex 'start_vertex' or 'end_vertex' is not in graph, add the vertex first.
        :param start_vertex: start vertex of the edge
        :param end_vertex: end vertex of the edge
        :return: NaN
        """
        if self.isContain(start_vertex.key):
            self.addVertex(start_vertex)
        if self.isContain(end_vertex.key):
            self.addVertex(end_vertex)
        start_vertex.addNeighbor(end_vertex)

    def getVertex(self, key):
        """
        Get vertex according to key
        :param key: key of the expected vertex
        :return:
        """
        vertex = self.vertexList.get(key)
        return vertex

    def isContain(self, key):
        """
        to determine whether the graph contains unit with 'key'
        :param key: key of the unit
        :return: boolean value, True for contain, False for not contain
        """
        if key not in self.vertexList.keys():
            return True
        else:
            return False

    def do_print(self):
        """
        Print arcText of the graph. Ignore the last vertex in the graph, which is used for conveniently constructing the entire CNN.
        :return: arcText string of the graph
        """
        print_str = ''
        keys = list(self.vertexList.keys())
        for i in range(0, len(keys) - 1):
            print_str = print_str + str(self.vertexList[keys[i]])
            if i != len(keys) - 2:
                print_str = print_str + '\n'
        return print_str

    def assign_id_for_graph(self, start_index):
        """
        assign 'id' and 'connect_to' information for units according to topological order of vertices(the matrix is upper triangular matrix)

        :param self: Graph object to update
        :param start_index: start index count of the graph
        :return: Graph object after assigning id
        """
        # initialize index list before and after update
        before_list = list(self.vertexList.keys())
        after_list = [start_index + i for i in range(0, len(before_list))]
        # construct an index dict for before and after list
        indexDict = {}
        for i in range(0, len(before_list)):
            indexDict[before_list[i]] = after_list[i]

        # assign the id and connect_to information in each unit
        for v in before_list:
            tempUnit = self.vertexList[v]
            tempUnit.id = indexDict[v]
            beforConnect = tempUnit.connect_to_list
            afterConnect = []
            for i in beforConnect:
                afterConnect.append(indexDict[i])
            tempUnit.connect_to_list = afterConnect
            tempUnit.connect_to = ",".join('%s' % id for id in afterConnect)

    def calculate_in_degree(self):
        """
        Calculate in_degree for all vertices for topological sorting.
        :return: NaN
        """
        for i in self.vertexList.keys():
            vertex = self.vertexList[i]
            connect_toList = vertex.connect_to_list
            for j in connect_toList:
                if j in self.indegreeDict.keys():
                    self.indegreeDict[j] = 1 + self.indegreeDict[j]
                else:
                    self.indegreeDict[j] = 1
        for i in self.vertexList.keys():
            if i not in self.indegreeDict.keys():
                self.indegreeDict[i] = 0

    def calculate_topo_dict(self):
        """
        Get topological order for vertices, and save it into topoDict
        :return: NaN
        """
        for i in self.vertexList.keys():
            j = 0
            while (j < self.vSize) & (self.indegreeDict[j] != 0):
                j = j + 1
            self.topoDict[i] = j
            self.indegreeDict[i] = -1
            for k in self.vertexList[i].connect_to_list:
                self.indegreeDict[k] = self.indegreeDict[k] - 1

    def find_longest_path(self):
        """
        Find longest path in the graph. If there are several path with same length, we compare the hash values of each
        vertex from the tail to the head in the path. If the hash values in paths are the same, we select it according
        to the topological order of the last vertex in the path (smaller topo order -> long).
        :returns: list of key in the path; whether the graph has unused vertex
        """

        maxPath_to_vertex = [0 for i in range(self.vSize)]  # max path to vertex

        path = [[0 for j in range(self.vSize)] for i in range(self.vSize)]  # max path from vertex to vertex

        for j in range(0, self.vSize):
            if self.vertexList[j].isUse == 0:
                v2 = self.topoDict[j]
                for k in range(0, j):
                    if self.vertexList[k].isUse == 0:
                        v1 = self.topoDict[k]
                        if v2 in self.vertexList[v1].connect_to_list:
                            if maxPath_to_vertex[v1] + 1 >= maxPath_to_vertex[v2]:
                                maxPath_to_vertex[v2] = maxPath_to_vertex[v1] + 1
                                path[v1][v2] = maxPath_to_vertex[v2]

        # max value of the path, which means the last vertex of the longest path
        last_vertex_value = max(maxPath_to_vertex)
        # find the longest path
        pathStack = []  # save vertex index in the path

        # if the longest path not equal to 0, find the last vertex index
        if last_vertex_value != 0:
            last_Vertex = -1
            for i in range(self.vSize):
                if self.vertexList[i].isUse == 0:
                    if maxPath_to_vertex[i] == last_vertex_value:
                        if last_Vertex == -1:
                            last_Vertex = i
                            maxHash = hashlib.sha224(self.vertexList[i].value.encode("utf8"))
                            self.vertexList[i].isUse = 1
                        else:
                            tempHash = hashlib.sha224(self.vertexList[i].value.encode("utf8"))
                            if tempHash.hexdigest() > maxHash.hexdigest():
                                self.vertexList[last_Vertex].isUse = 0
                                maxHash = tempHash
                                last_Vertex = i
                                self.vertexList[last_Vertex].isUse = 1
            i = last_Vertex
            # Search forward from the end node according to the topological order to find vertices of the longest path.
            while i > 0:
                points = [0] * i
                for j in range(i):
                    points[j] = path[j][i]
                points.sort()
                maxValue = points[-1]  # find the max path from other vertex to vertex i
                if maxValue != 0:
                    target_j = -1
                    maxHash = ''
                    for j in range(i):
                        # if there are several vertex with same maxValue, select one according to their hash value
                        if path[j][i] == maxValue:
                            if target_j == -1:
                                target_j = j
                                maxHash = hashlib.sha224(self.vertexList[j].value.encode("utf8"))
                                self.vertexList[j].isUse = 1
                            else:
                                tempHash = hashlib.sha224(self.vertexList[j].value.encode("utf8"))
                                if tempHash.hexdigest() > maxHash.hexdigest():
                                    maxHash = tempHash
                                    self.vertexList[target_j].isUse = 0
                                    target_j = j
                                    self.vertexList[j].isUse = 1
                    pathStack.append(i)
                    i = target_j
                    if maxPath_to_vertex[i] == 0:
                        pathStack.append(i)
                        break

        # if there is no path in the graph, we order all vertices according to their hash value
        if len(pathStack) == 0:
            sortList = []
            for i in range(self.vSize):
                if self.vertexList[i].isUse == 0:
                    sortList.append(i)
                    self.vertexList[i].isUse = 1

            for i in range(len(sortList)):
                for j in range(0, len(sortList) - i):
                    maxHash_i = hashlib.sha224(self.vertexList[i].value.encode("utf8"))
                    maxHash_j = hashlib.sha224(self.vertexList[j].value.encode("utf8"))
                    if maxHash_i.hexdigest() < maxHash_j.hexdigest():
                        tempi = sortList[i]
                        sortList[i] = sortList[j]
                        sortList[j] = tempi
            pathStack = sortList
            return pathStack, False

        # the pathStack is reverse order, we need to reverse it to get longest path vertices
        pathStack.reverse()
        return pathStack, True

    def get_ordered_arcText(self):
        """
        Order the arcText units according to the Algorithm 2 (find longest path).
        :return: ordered arcText string
        """
        id_list = []
        self.calculate_in_degree()
        self.calculate_topo_dict()
        temp_list, isContinue = self.find_longest_path()
        id_list.extend(temp_list)
        # execute until all vertices are ordered
        while isContinue == True:
            temp_list, isContinue = self.find_longest_path()
            id_list.extend(temp_list)

        # define two dict to save the relationship between id before order and after order
        after_before_dict = dict(zip([i for i in range(len(id_list))], id_list))
        before_after_dict = dict(zip(id_list, [i for i in range(len(id_list))]))

        result_arcText = ''

        # assign value to result_arcText
        for i in range(len(after_before_dict)):
            newConnectTo = []
            connectTo = self.vertexList[after_before_dict[i]].connect_to_list
            for item in connectTo:
                newConnectTo.append(str(before_after_dict[item]))
            result_arcText = result_arcText + 'id:' + str(i) + ';' + self.vertexList[
                after_before_dict[i]].value + 'connect_to:' + ','.join(newConnectTo)
            if i < len(before_after_dict) - 1:
                result_arcText = result_arcText + '\n'

        return result_arcText
