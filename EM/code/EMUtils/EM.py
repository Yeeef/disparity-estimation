from .config import *
from . import utils
from . import Estep
from . import Mstep
import os

class EM(object):
    def __init__(self, save_dir):
        # utils.init_param_with_truth()
        if not os.path.exists(save_dir):
            print(f"cannot find path: {save_dir}")
        utils.init_param()
        self.iter = 0
        self.save_dir = save_dir
        # print(f"free energy: {Estep.free_energy()}")

    def estimation(self):
        Estep.E_step()
    
    def maxmization(self):
        Mstep.M_step()

    def e_and_m(self):
        self.estimation()
        print(f"\n {self.iter} E step finished")
        self.maxmization()
        print(f"\n {self.iter} M step finished")
        self.save_fig()
        # print(f"free energy: {Estep.free_energy()}")
        
        print(f"\n {self.iter} save fig successfully in {self.save_dir}")
        self.iter += 1

    def save_fig(self):
        disp_fig = Image.fromarray(FACTOR * disparity_image.astype('uint8'), "L")
        visi_fig = Image.fromarray(255 * visible_image.astype('uint8'), "L")
        ideal_fig = Image.fromarray(ideal_image.astype('uint8'), "L")
        disp_fig.save(os.path.join(self.save_dir, "disp_fig" + str(self.iter) + ".png"), quality=100)
        visi_fig.save(os.path.join(self.save_dir, "visi_fig" + str(self.iter) + ".png"), quality=100)
        ideal_fig.save(os.path.join(self.save_dir, "ideal_fig" + str(self.iter) + ".png"), quality=100)