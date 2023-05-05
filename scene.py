import numpy as np
import taichi as ti
import taichi.math as tm

from util import *
from enum import Enum

vec2 = tm.vec2
vec3 = tm.vec3

@ti.dataclass
class BoxState:
    p : vec2  # position (center of mass)
    q : vec2  # orientation (cosine/sine pair)
    v : vec2  # linear velocity
    ω : float # angular velocity
    l : vec2  # dimensions of box
    m : float # mass
    I : float # moment of inertia
    rad: float # collision detection radius

@ti.dataclass
class Boundary:
    p: vec2
    n: vec2
    rad:float

class Init(Enum):
    STACK = "stack"
    BW_C = "box_wall_collide"
    BB_C = "box_box_collide"
    BC_C = "box_corner_collide"
    RANDOM = "random_box"


# Scene-related Data
@ti.data_oriented
class Scene:
    def __init__(self, initial_state, dt):
        self.initial_state = initial_state
        self.dt = dt
        self.ρ = 1
        self.startScene()

    def startScene(self):
        if self.initial_state == Init.STACK:
            self.init_box_stack()
        elif self.initial_state == Init.BW_C:
            self.init_box_wall_collision()
        elif self.initial_state == Init.BB_C:
            self.init_box_bounce()
        elif self.initial_state == Init.BC_C:
            self.init_box_corner_collision()
        elif self.initial_state == Init.RANDOM:
            self.init_random_boxes()

        self.init_box_corners_normals()
        self.init_params()
        self.init_indices()


    def init_box_corners_normals(self):
        if not hasattr(self, "corners"):
            # array with the signs of the local coords of the 4 corners of a box
            self.corners = ti.Vector.field(2, shape=(4,), dtype=ti.f32)
            self.corners.from_numpy(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32))

        if not hasattr(self, "normals"):
            # array with the local normals of the 4 edges of a box
            self.normals = ti.Vector.field(2, shape=(4,), dtype=ti.f32)
            self.normals.from_numpy(np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.float32))


    def init_square_boundaries(self):
        self.b_eps = 1e-1
        if not hasattr(self, "boundaries"):
            self.boundaries = Boundary.field(shape=(4,))
            self.boundaries.p.from_numpy(np.array([
                [self.b_eps, 1 - self.b_eps],
                [self.b_eps, self.b_eps],
                [1 - self.b_eps, self.b_eps],
                [1 - self.b_eps, 1 - self.b_eps]
            ], dtype=np.float32))
            self.boundaries.n.from_numpy(np.array([
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1]
            ], dtype=np.float32))
            self.boundaries.rad.from_numpy(np.ones(4, dtype=np.float32) * 1e-3)
            self.boundary_indices = ti.field(shape=(8,), dtype=ti.i32)


    def init_random_boxes(self):
        self.init_square_boundaries()

        # Stratified Sampling of random boxes
        strat_x = 7
        self.g = vec2(0, -.2)
        self.N = strat_x ** 2
        self.rad = 1e-3

        spacing = (1 - 2 * self.b_eps) / strat_x
        x = np.linspace(self.b_eps * 1, 1 - 1 * self.b_eps, strat_x + 1)[:-1] + spacing * 0.5
        y = np.linspace(self.b_eps * 1, 1 - 1 * self.b_eps, strat_x + 1)[:-1] + spacing * 0.5
        xgrid, ygrid = np.meshgrid(x, y)
        box_cs = np.stack([xgrid.flatten(), ygrid.flatten()], axis=-1).astype(np.float32)
        self.hMin, self.hMax = 0.2 * spacing, 0.8 * spacing
        vm = 2.0 * spacing * 0.5
        ωm = 1.0

        self.init_boxes()
        rand_local = (np.random.random((self.N, 2)).astype(np.float32) - 0.5) * spacing * 0.5
        self.boxes.p.from_numpy(box_cs + rand_local)
        th = 2 * np.pi * np.random.random((self.N,)).astype(np.float32)
        self.boxes.q.from_numpy(np.stack([np.cos(th), np.sin(th)], axis=-1).astype(np.float32))
        self.boxes.v.from_numpy(np.random.normal(0, vm, (self.N, 2)).astype(np.float32))
        self.boxes.ω.from_numpy(np.random.normal(0, ωm, (self.N,)).astype(np.float32))
        self.boxes.l.from_numpy(
            (self.hMin + np.random.random((self.N, 2)) * (self.hMax - self.hMin)).astype(np.float32))
        self.boxes.rad.from_numpy(np.ones(self.N, dtype=np.float32) * self.rad)

    def init_box_stack(self):
        self.init_square_boundaries()

        self.g = vec2(0, -.2)
        self.N = 5
        l = vec2(0.5, 0.09)
        ratio = 0.8
        eps = l.y * 0.2
        rad = 1e-3

        self.init_boxes()
        scales = np.power(ratio, np.arange(self.N))
        ls = np.outer(scales, l)
        ys = np.cumsum(ls[:, 1]) - ls[0, 1] / 2 + eps * np.arange(1, self.N + 1) + self.b_eps
        xs = np.ones(self.N) * 0.5
        self.boxes.p.from_numpy(np.stack([xs, ys], axis=-1).astype(np.float32))
        self.boxes.q.from_numpy(np.stack([np.ones(self.N), np.zeros(self.N)], axis=-1).astype(np.float32))
        self.boxes.v.from_numpy(np.zeros((self.N, 2)).astype(np.float32))
        self.boxes.ω.from_numpy(np.zeros(self.N).astype(np.float32))
        self.boxes.l.from_numpy(ls.astype(np.float32))
        self.boxes.rad.from_numpy(np.ones(self.N,dtype=np.float32) * rad)

    def init_box_wall_collision(self):
        self.init_square_boundaries()

        self.g = vec2(0, 0)
        self.N = 1
        l = vec2(0.2, 0.2)
        rad = 1e-3

        self.init_boxes()
        ls = np.outer(np.ones(self.N, ), l)
        ys = 0.5
        xs = self.b_eps * 2 + l.x / 2
        self.boxes.p.from_numpy(np.array([[xs, ys]], dtype=np.float32))
        self.boxes.q.from_numpy(np.array([[1, 0]], dtype=np.float32))
        self.boxes.v.from_numpy(np.array([[0.2, 0]], dtype=np.float32))
        self.boxes.ω.from_numpy(np.array([0.1], dtype=np.float32))
        self.boxes.l.from_numpy(ls)
        self.boxes.rad.from_numpy(np.ones(self.N, dtype=np.float32) * rad)

    def init_box_corner_collision(self):
        self.init_square_boundaries()

        self.g = vec2(0, 0)
        self.N = 1
        l = vec2(0.2, 0.2)
        rad = 1e-3

        self.init_boxes()
        ls = np.outer(np.ones(self.N, ), l)
        ys = l.y * np.sqrt(2) * 0.5 + self.b_eps
        xs = ys
        self.boxes.p.from_numpy(np.array([[xs, ys]], dtype=np.float32))
        self.boxes.q.from_numpy(np.array([[np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)]], dtype=np.float32))
        self.boxes.v.from_numpy(np.array([[0.2, 0.2]], dtype=np.float32))
        self.boxes.ω.from_numpy(np.array([0], dtype=np.float32))
        self.boxes.l.from_numpy(ls)
        self.boxes.rad.from_numpy(np.ones(self.N, dtype=np.float32) * rad)

    def init_box_bounce(self):
        self.init_square_boundaries()

        self.g = vec2(0, 0)
        self.N = 2
        l = vec2(0.2, 0.2)
        eps = l.y * (np.sqrt(2) - 1 + 0.1)
        rad = 1e-3

        self.init_boxes()
        ls = np.outer(np.ones(self.N, ), l)
        ys = np.cumsum(ls[:, 1]) - ls[0, 1] / 2 + eps * np.arange(1, self.N + 1) + self.b_eps
        xs = np.ones(self.N) * 0.5
        self.boxes.p.from_numpy(np.stack([xs, ys], axis=-1).astype(np.float32))
        self.boxes.q.from_numpy(
            np.stack([np.ones(self.N) * np.cos(np.pi / 4.0), np.ones(self.N) * np.sin(np.pi / 4.0)], axis=-1).astype(np.float32))
        self.boxes[0].q = vec2(1, 0)
        self.boxes.v.from_numpy(np.stack([np.zeros(self.N), np.ones(self.N) * -0.2], axis=-1).astype(np.float32))
        self.boxes[0].v = vec2(0, 0)
        self.boxes.ω.from_numpy(np.zeros(self.N).astype(np.float32))
        self.boxes.l.from_numpy(ls)
        self.boxes.rad.from_numpy(np.ones(self.N).astype(np.float32) * rad)

    def init_boxes(self):
        if not hasattr(self, "boxes"):
            print("init new field with N ", self.N)
            self.boxes = BoxState.field(shape=(self.N,))
            self.vertices = ti.Vector.field(2, shape=(4*self.N,), dtype=ti.f32)
            self.indices = ti.field(shape=(6 * self.N,), dtype=ti.i32)
            self.edge_indices = ti.field(shape=(8 * self.N,), dtype=ti.i32)

    @ti.kernel
    def init_params(self):
        for i in self.boxes:
            box = self.boxes[i]
            self.boxes[i].m = self.ρ * box.l.x * box.l.y
            self.boxes[i].I = (1 / 12) * self.boxes[i].m * box.l.dot(box.l)

    @ti.kernel
    def init_indices(self):
        for i in range(4):
            self.boundary_indices[2 * i] = i
            self.boundary_indices[2 * i + 1] = (i + 1) % 4

        for i in range(self.N):
            self.indices[6 * i + 0] = 4 * i + 0
            self.indices[6 * i + 1] = 4 * i + 1
            self.indices[6 * i + 2] = 4 * i + 2
            self.indices[6 * i + 3] = 4 * i + 0
            self.indices[6 * i + 4] = 4 * i + 2
            self.indices[6 * i + 5] = 4 * i + 3

            for j in range(4):
                self.edge_indices[8 * i + j * 2] = 4 * i + j
                self.edge_indices[8 * i + j * 2 + 1] = 4 * i + (j + 1) % 4

    @ti.kernel
    def compute_KE(self)->float:
        KE = 0.0

        for i in self.boxes:
            box = self.boxes[i]
            KE += 0.5 * box.v.dot(box.v) * box.m + 0.5 * box.ω * box.ω * box.I

        return KE

    @ti.kernel
    def copy_vertices(self):
        for i in self.boxes:
            box = self.boxes[i]
            for j in range(4):
                s = self.corners[j]
                self.vertices[4 * i + j] = b2w(box.p, box.q, s * box.l / 2)

    # update velocity
    @ti.kernel
    def update_vel(self):
        for i in self.boxes:
            self.boxes[i].v += self.dt * self.g 
            

    # update position
    @ti.kernel
    def update_posn(self):
        for i in self.boxes:
            self.boxes[i].p += self.dt * self.boxes[i].v
            direciton = self.boxes[i].q + self.dt * cross(self.boxes[i].ω, self.boxes[i].q)
            self.boxes[i].q = tm.normalize(direciton)

    @ti.kernel
    def print_boxes(self):
        for i in self.boxes:
            box = self.boxes[i]
            print(f"box {i} p {box.p} q {box.q}")




