import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from torch.utils.data import Dataset


class FGL4Dataset_Rept(Dataset):
    all_list = []

    def __init__(self, point_source_catalogue, wanted_type_col, other_cols):
        self.all_list = []
        print(len(point_source_catalogue.data))
        for source in point_source_catalogue.data:
            lst = [source[0].split(' ')[1]]
            lst.append(source[wanted_type_col])
            for col in other_cols:
                if source[col] is not None and source[col] != '':
                    lst.append(source[col].split()[1])
                else:
                    lst.append('')
            self.all_list.append(lst)

    def get_all_list(self):
        return self.all_list
