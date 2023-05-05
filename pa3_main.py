import numpy as np
import taichi as ti
import taichi.math as tm

from rigid_sim import *
from GUI import *

width = 768

dt = 1/60
μ = 0

# turn on the parameters below on for the
# box stacking test and random box test
# β = 20
# Cr = 0.7

# use the parameters below for all other tests
β = 1.0
Cr = 1.0

ti.init(arch=ti.cpu)
inits = [Init.BW_C, Init.BC_C, Init.BB_C, Init.STACK, Init.RANDOM]

rigidSims = {init:SimGUI(RigidSim(init, dt, β, Cr, μ)) for init in inits}
gui = A3GUI(width, rigidSims, Init.BW_C, )
# gui = A3GUI(width, rigidSims, Init.BW_C, record=True)

gui.startGUI()
while gui.is_window_running():
    gui.currSim.scene.update_vel()

    # collision detection
    gui.currSim.collision.clearCollision()
    gui.currSim.collision.collide_bounds()
    gui.currSim.collision.collide_all()

    # collision response
    prev_KE = gui.currSim.scene.compute_KE()
    gui.currSim.collision.response.PGS()
    gui.currSim.collision.response.apply_impulses()
    after_KE = gui.currSim.scene.compute_KE()
    if abs(Cr - 1) <= 1e-4 and abs(β) <= 1e-4 and abs(prev_KE - after_KE) >= 1e-5:
        print(f"the energy before applying impulse {prev_KE}" +
              f" and the energy after applying impulse {after_KE} are significantly different")

    # symplectic euler update position
    gui.currSim.scene.update_posn()

    # GUI
    gui.drawUI()

    gui.currSim.scene.copy_vertices()
    gui.currSim.set_colors()
    gui.drawCanvas()
    gui.update_iter()

gui.endGUI()



