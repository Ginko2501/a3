import numpy as np
import taichi as ti
import taichi.math as tm

from scene import *
from response import *
from collision import *

@ti.data_oriented
class RigidSim:
    def __init__(self, initial_state, dt, β, Cr, μ):
        self.initial_state = initial_state
        self.dt = dt
        self.β = β
        self.Cr = Cr
        self.μ = μ

        self.init_rigid_sim()


    def init_rigid_sim(self):
        self.scene = Scene(self.initial_state, self.dt)
        self.response = CollisionResponse(self.scene, self.Cr, self.β, self.μ)
        self.collision = Collision(self.scene, self.response)
