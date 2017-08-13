import pandas as pd
import numpy as np


class TimeWeighter(object):
    def __init__(self):
        self.one_month_weight_decay = 0.5