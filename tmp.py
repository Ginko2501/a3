import taichi as ti

ti.init(arch=ti.cpu)

M = ti.field(dtype=ti.f32, shape=(2, 3))

M2 = M[:, 1:]

print(M2)