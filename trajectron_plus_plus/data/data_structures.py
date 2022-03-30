from collections import OrderedDict

import numpy as np


class DoubleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        """
        Data Structure mirroring some functionality of double indexed pandas DataFrames.
        Indexing options are:
        [:, (header1, header2)]
        [:, [(header1, header2), (header1, header2)]]
        [:, {header1: [header21, header22]}]

        A SingleHeaderNumpyArray can is returned if an element of the first header is querried as attribut:
        doubleHeaderNumpyArray.position -> SingleHeaderNumpyArray

        :param data: The numpy array.
        :param header: The double header structure as list of tuples [(header11, header21), (header11, header22) ...]
        """
        self.data = data
        self.header = header
        self.double_header_lookup = OrderedDict()
        self.tree_header_lookup = OrderedDict()
        for i, header_item in enumerate(header):
            self.double_header_lookup[header_item] = i
            if header_item[0] not in self.tree_header_lookup:
                self.tree_header_lookup[header_item[0]] = dict()
            self.tree_header_lookup[header_item[0]][header_item[1]] = i

    def __mul__(self, other):
        return DoubleHeaderNumpyArray(self.data * other, self.header)

    def get_single_header_array(self, h1: str, rows=slice(None, None, None)):
        data_integer_indices = list()
        h2_list = list()
        for h2 in self.tree_header_lookup[h1]:
            data_integer_indices.append(self.tree_header_lookup[h1][h2])
            h2_list.append(h2)
        return SingleHeaderNumpyArray(self.data[rows, data_integer_indices], h2_list)

    def __getitem__(self, item):
        rows, columns = item
        data_integer_indices = list()
        if type(columns) is dict:
            for h1, h2s in columns.items():
                for h2 in h2s:
                    data_integer_indices.append(self.double_header_lookup[(h1, h2)])
            return self.data[rows, data_integer_indices]
        elif type(columns) is list:
            for column in columns:
                assert type(column) is tuple, "If Index is list it hast to be list of double header tuples."
                data_integer_indices.append(self.double_header_lookup[column])
            return self.data[rows, data_integer_indices]
        elif type(columns) is tuple:
            return self.data[rows, self.double_header_lookup[columns]]
        else:
            assert type(item) is str, "Index must be str, list of tuples or dict of tree structure."
            return self.get_single_header_array(item, rows=rows)

    def __getattr__(self, item):
        if not item.startswith('_'):
            if item in self.tree_header_lookup.keys():
                return self.get_single_header_array(item)
            else:
                try:
                    return self.data.__getattribute__(item)
                except AttributeError:
                    return super().__getattribute__(item)
        else:
            return super().__getattribute__(item)


class SingleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        self.data = data
        self.header_lookup = OrderedDict({h: i for i, h in enumerate(header)})

    def __getitem__(self, item):
        rows, columns = item
        data_integer_indices = list()
        if type(columns) is list or type(columns) is tuple:
            for column in columns:
                data_integer_indices.append(self.header_lookup[column])
        else:
            data_integer_indices = self.header_lookup[columns]
        return self.data[rows, data_integer_indices]

    def __getattr__(self, item):
        if not item.startswith('_'):
            if item in self.header_lookup.keys():
                return self[:, item]
            else:
                try:
                    return self.data.__getattribute__(item)
                except AttributeError:
                    return super().__getattribute__(item)
        else:
            return super().__getattribute__(item)
