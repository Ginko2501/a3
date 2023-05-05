import numpy as np
import taichi as ti
import taichi.math as tm
from rigid_sim import *

vec2 = tm.vec2
vec3 = tm.vec3

@ti.data_oriented
class SimGUI:
    def __init__(self, sim):
        self.collision = sim.collision
        self.scene = sim.scene
        self.colors = ti.Vector.field(3, shape=(4 * self.scene.N,), dtype=ti.f32)
        self.tri_colors = ti.Vector.field(3, shape=(4 * self.scene.N,), dtype=ti.f32)
        self.widths = ti.field(shape=(4 * self.scene.N,), dtype=ti.f32)
        self.widths.fill(0.1)

        self.init_random_colors()

    @ti.kernel
    def init_random_colors(self):
        for i in range(self.scene.N):
            self.tri_colors[i * 4] = vec3(ti.random(float) * 0.5 + 0.5,
                                          ti.random(float) * 0.5 + 0.5,
                                          ti.random(float) * 0.5 + 0.5)
        for i in self.colors:
            self.tri_colors[i] = self.tri_colors[i // 4 * 4]

    @ti.kernel
    def set_colors(self):
        for i in self.scene.vertices:
            self.colors[i] = tm.vec3(self.collision.coll[i // 4] / 255.0)

    def restart(self):
        self.scene.startScene()

@ti.data_oriented
class A3GUI:
    def __init__(self, width, simGUIs, INITTEST, record=False):
        self.win_width = width
        self.simGUIs = simGUIs
        assert(len(simGUIs) != 0)
        self.currSim = simGUIs[INITTEST]
        self.record = record
        self.iter = 0

        if self.record:
            self.video_manager = ti.tools.VideoManager(output_dir="./output", framerate=60, automatic_build=False)

    def startGUI(self):
        # Create Taichi UI
        self.window = ti.ui.Window("Rigid bodies...", (self.win_width, self.win_width), vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((0.3, 0.3, 0.3))

    def is_window_running(self):
        if self.record:
            return self.window.running and self.iter < 300
        else:
            return self.window.running

    def update_iter(self):
        self.iter += 1

    def drawUI(self):
        gui = self.window.get_gui()
        name = "Tests"
        x, y, width, height = 0.02, 0.02, 0.20, 0.20
        with gui.sub_window(name, x, y, width, height):
            is_bw = gui.button(Init.BW_C.value)
            is_bb = gui.button(Init.BB_C.value)
            is_bc = gui.button(Init.BC_C.value)
            is_stack = gui.button(Init.STACK.value)
            is_random = gui.button(Init.RANDOM.value)

            hash = None

            if is_bw:
                hash = Init.BW_C
            elif is_bc:
                hash = Init.BC_C
            elif is_bb:
                hash = Init.BB_C
            elif is_stack:
                hash = Init.STACK
            elif is_random:
                hash = Init.RANDOM

            if hash is not None:
                self.simGUIs[hash].restart()
                self.iter = 0
                self.currSim = self.simGUIs[hash]

    def drawCanvas(self):
        self.canvas.lines(self.currSim.scene.vertices, width=0.004, indices=self.currSim.scene.edge_indices, per_vertex_color=self.currSim.colors)
        self.canvas.lines(self.currSim.scene.boundaries.p, width=0.002, indices=self.currSim.scene.boundary_indices, color=(1.0, 1.0, 0.0))
        if self.currSim.scene.initial_state == Init.RANDOM:
            self.canvas.triangles(self.currSim.scene.vertices,
                                  indices=self.currSim.scene.indices,
                                  per_vertex_color=self.currSim.tri_colors)
        else:
            self.canvas.circles(self.currSim.collision.collPs, radius=0.01, color=(1, 0, 1))


        if self.record and self.iter % 1 == 0:
            img = self.window.get_image_buffer_as_numpy()
            self.video_manager.write_frame(img)
        self.window.show()

    def endGUI(self):
        if self.record:
            self.video_manager.make_video(gif=False, mp4=True)



