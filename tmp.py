import taichi as ti
import taichi.math as tm
import numpy as np

vec2i = ti.types.vector(2, ti.i32)

width = 768
N = 5
r = 0.2 / np.sqrt(N)
vm = 5 * r
m = 1

ti.init(arch=ti.cpu)#, cpu_max_num_threads=1)

x = ti.Vector.field(2, shape=(N,), dtype=ti.f32)
v = ti.Vector.field(2, shape=(N,), dtype=ti.f32)
coll = ti.field(shape=(N,), dtype=ti.u8)
colors = ti.Vector.field(3, shape=(N,), dtype=ti.f32)

x.from_numpy(np.random.random((N,2)).astype(np.float32))
v.from_numpy(np.random.normal(0, vm, (N,2)).astype(np.float32))

h = 1/60

@ti.kernel
def update():

    # E_k = 0.0
    # for i in v:
    #     E_k += 0.5 * m * v[i].dot(v[i])
    # print("energy", E_k)

    for i in coll:
        coll[i] = ti.u8(tm.floor(0.9 * coll[i]))

    # This is currently a problem; position updates will cause CCD
    # to miss collisions
    for i in x:
        if x[i].x < r:
            x[i].x = 2*r - x[i].x
            v[i].x = -v[i].x
        if x[i].x > 1.0 - r:
            x[i].x = 2*(1.0 - r) - x[i].x
            v[i].x = -v[i].x
        if x[i].y < r:
            x[i].y = 2*r - x[i].y
            v[i].y = -v[i].y
        if x[i].y > 1.0 - r:
            x[i].y = 2*(1.0 - r) - x[i].y
            v[i].y = -v[i].y

# list of collisions
C_pairs = ti.Vector.field(2, ti.i32, (1,))
C_times = ti.field(ti.f32, (1,))


@ti.func
def sphere_ccd(x0, x1, v0, v1, tMin, tMax):
    d0 = x0 - x1
    vrel = v0 - v1
    t_c = tMax
    a, b, c = vrel.dot(vrel), 2*vrel.dot(d0), d0.dot(d0) - 4*r*r
    D = b*b - 4*a*c
    if D >= 0:
        r = -0.5 * (b + tm.sign(b) * tm.sqrt(D))
        ta, tb = r/a, c/r
        if tMin <= ta <= t_c:
            t_c = ta
        if tMin <= tb <= t_c:
            t_c = tb
    return t_c if t_c < tMax else np.nan

@ti.kernel
def cd_brute(h : ti.f32) -> ti.f32:
    t_c = h
    ti.loop_config(serialize=True)
    for i in range(N):
        for j in range(i):
            t_c = sphere_ccd(x[i],  x[j], v[i], v[j], 0, h)
            if not tm.isnan(t_c):
                m_i, m_j = m, m
                if (v[i] - v[j]).dot(x[i] - x[j]) < 0:
                    coll[i] = ti.u8(255)
                    coll[j] = ti.u8(255)
                    n = tm.normalize(x[i] - x[j])
                    v_n = (v[i] - v[j]).dot(n)
                    m_eff = 1/(1/m_i + 1/m_j)
                    gamma = -2 * m_eff * v_n
                    impulse = gamma * n
                    v[i] = v[i] + impulse / m_i
                    v[j] = v[j] - impulse / m_j
            if tm.length(x[i] - x[j]) < 2*r:
                print("yikes! particles overlap!", i, j)

    return t_c


@ti.kernel
def advance(t : ti.f32):
    for k in x:
        x[k] = x[k] + t * v[k]


@ti.kernel
def set_colors():
    for i in x:
        colors[i] = tm.vec3(coll[i] / 255.0)


# Create Taichi UI
window = ti.ui.Window("Collisions!", (width, width), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 0.5))

while window.running:

    update()
    cd_brute(h)
    advance(h)
    set_colors()

    # radius needs to be doubled on high DPI displays
    canvas.circles(x, radius=2*r, per_vertex_color=colors)

    window.show()
