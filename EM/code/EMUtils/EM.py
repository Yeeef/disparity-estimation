from .config import *
from . import utils
from . import Estep
from . import Mstep

class EM(object):
    def __init__(self):
        utils.init_param()

    def estimation(self):
        Estep.E_step()
    
    def maxmization(self):
        Mstep.M_step()

    def e_and_m(self):
        self.estimation()
        self.maxmization()

    def visualize(self):
        pass