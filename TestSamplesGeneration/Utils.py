from enum import Enum


class Frequency(Enum):
    m5 = 'm5'
    m1 = 'm1'

    def get_frequency_part_in_hours(self):
        return int(str(self.value)[1])/60

class Tickers(Enum):
    ALRS = 'ALRS'
    BRX = 'BRX'
    GAZP = 'GAZP'
    GZX = 'GZX'
    LKOH = 'LKOH'
    MGNT = 'MGNT'
    MICEXINDEXCF = 'MICEXINDEXCF'
    MOEX = 'MOEX'
    ROSN = 'ROSN'
    SBER = 'SBER'
    SBERP = 'SBERP'
    SIX = 'SIX'
    SNGS = 'SNGS'
    USD000UTSTOM = 'USD000UTSTOM'
    VTBR = 'VTBR'