from .config import *
from . import utils
from . import Estep
from . import Mstep
from . import MstepFast
from . import EstepFast
import os
import time

class EM(object):
    def __init__(self, save_dir):
        # utils.init_param_with_truth()
        if not os.path.exists(save_dir):
            print(f"cannot find path: {save_dir}")
            print("fuck")
        utils.init_param()
        disp_fig = Image.fromarray(FACTOR * disparity_image.astype('uint8'), "L")
        visi_fig = Image.fromarray(255 * visible_image.astype('uint8'), "L")
        disp_fig.save(os.path.join(save_dir, "disp_fig_init" + ".png"), quality=100)
        visi_fig.save(os.path.join(save_dir, "visi_fig_init" + ".png"), quality=100)
        self.iter = 0
        self.save_dir = save_dir
        # print(f"free energy: {Estep.free_energy()}")

    def estimation(self):
        # Estep.E_step()
        EstepFast.E_step_fast()
    
    def maxmization(self):
        # Mstep.M_step()
        MstepFast.M_step_fast()

    def e_and_m(self):
        start = time.time()
        self.estimation()
        print(f"\n {self.iter} E step finished")
        self.maxmization()
        print(f"\n {self.iter} M step finished")
        self.save_fig()
        # print(f"free energy: {Estep.free_energy()}")
        
        print(f"\n {self.iter} save fig successfully in {self.save_dir}")
        print(f"Time elapsed: {time.time() - start} s")
        self.iter += 1

    def save_fig(self):
        disp_fig = Image.fromarray(FACTOR * disparity_image.astype('uint8'), "L")
        visi_fig = Image.fromarray(255 * visible_image.astype('uint8'), "L")
        ideal_fig = Image.fromarray(ideal_image.astype('uint8'), "L")
        disp_fig.save(os.path.join(self.save_dir, "disp_fig" + str(self.iter) + ".png"), quality=100)
        visi_fig.save(os.path.join(self.save_dir, "visi_fig" + str(self.iter) + ".png"), quality=100)
        ideal_fig.save(os.path.join(self.save_dir, "ideal_fig" + str(self.iter) + ".png"), quality=100)