"""
This file define several units, which is used to represent arcText Unit and  vertex in the graph.
"""
class unit():
    def __init__(self, key):
        self.key = key
        self.connect_to_list = []
        self.value = ''
        self.isUse = 0

    def getConnections(self):
        return self.connect_to_list

    def addNeighbor(self, nbr):
        self.connect_to_list.append(nbr.key)

class conv_arcunit(unit):
    def __init__(self, key):
        unit.__init__(self, key)
        self.id = ''
        self.in_size=''
        self.out_size=''
        self.kernel=''
        self.stride='1-1'
        self.padding=''
        self.dilation='1'
        self.groups='1'
        self.bias_used='No'
        self.connect_to=''
    def __str__(self):
        return 'id:%s;in_size:%s;out_size:%s;kernel:%s;stride:%s;padding:%s;dilation:%s;groups:%s;bias_used:%s;connect_to:%s' % (
        self.id, self.in_size, self.out_size, self.kernel, self.stride, self.padding, self.dilation, self.groups,
        self.bias_used, self.connect_to)

    def concat_str(self):
        return 'in_size:%s;out_size:%s;kernel:%s;stride:%s;padding:%s;dilation:%s;groups:%s;bias_used:%s' % (
        self.in_size, self.out_size, self.kernel, self.stride, self.padding, self.dilation, self.groups,
        self.bias_used)


class pool_arcunit(unit):
    def __init__(self, key):
        unit.__init__(self, key)
        self.id=''
        self.type = 'Max'
        self.in_size = ''
        self.out_size = ''
        self.kernel = '3-3'
        self.stride = '1-1'
        self.padding = '1-1-1-1'
        self.dilation = '1'
        self.bias_used = 'No'
        self.connect_to = ''

    def __str__(self):
        return 'id:%s;type:%s;in_size:%s;out_size:%s;kernel:%s;stride:%s;padding:%s;dilation:%s;bias_used:%s;connect_to:%s' % (
        self.id, self.type, self.in_size, self.out_size, self.kernel, self.stride, self.padding, self.dilation,
        self.bias_used, self.connect_to)

    def concat_str(self):
        return 'type:%s;in_size:%s;out_size:%s;kernel:%s;stride:%s;padding:%s;dilation:%s;bias_used:%s' % (
            self.type, self.in_size, self.out_size, self.kernel, self.stride, self.padding, self.dilation,
            self.bias_used)


class full_arcunit(unit):
    def __init__(self, key):
        unit.__init__(self, key)
        self.id = ''
        self.in_size = ''
        self.out_size = ''
        self.connect_to = ''

    def __str__(self):
        return 'id:%s;in_size:%s;out_size:%s;connect_to:%s' % (self.id,self.in_size,self.out_size,self.connect_to)

    def concat_str(self):
        return 'in_size:%s;out_size:%s' % (self.in_size,self.out_size)



class mf_arcunit(unit):
    def __init__(self, key):
        unit.__init__(self, key)
        self.id = ''
        self.name = ''
        self.in_size = ''
        self.out_size = ''
        self.value = ''
        self.connect_to = ''

    def __str__(self):
        return 'id:%s;name:%s;in_size:%s;out_size:%s;value:%s;connect_to:%s' % (
        self.id, self.name, self.in_size, self.out_size, self.value, self.connect_to)

    def concat_str(self):
        return 'name:%s;in_size:%s;out_size:%s;value:%s' % (
        self.name, self.in_size, self.out_size, self.value)


if __name__ == "__main__":
    c = conv_arcunit()
    print('conv %s' %c)
    p = pool_arcunit()
    print('pool %s' %p)
    f = full_arcunit()
    print('full %s' %f)
    m = mf_arcunit()
    print('mf %s' %m)
