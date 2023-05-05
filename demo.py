
import taichi as ti
import taichi.math as tm
import numpy as np

vec2i = ti.types.vector(2, ti.i32)
vec2 = tm.vec2

width = 768
N = 100
r = 0.4 / np.sqrt(N)
vm = 5 * r
m = 1
g = vec2(0, -0.1)  # acceleration of gravity
c_r = .7   # coefficient of restitution

ti.init(arch=ti.cpu)#, cpu_max_num_threads=1, debug=True)

x = ti.Vector.field(2, shape=(N,), dtype=ti.f32)
v = ti.Vector.field(2, shape=(N,), dtype=ti.f32)

if N == 3:
    # test case with 3 stacked particles
    x.from_numpy(np.array([
        [0.2, 0.2], [0.8, 0.2], [0.5, 0.50]
        ]).astype(np.float32))
    r = 0.2
elif N == 6:
    # test case with 6 stacked particles
    x.from_numpy(np.array([
        [0.15, 0.15], [0.5, 0.15], [.85, .15],
        [0.325, 0.46], [0.675, 0.46],
        [0.5, 0.8]
        ]).astype(np.float32))
    r = 0.15
else:
    # randomly initialized particle positions and velocities
    x.from_numpy(np.random.random((N,2)).astype(np.float32))
    v.from_numpy(np.random.normal(0, vm, (N,2)).astype(np.float32))

h = 1/60       # timestep for integration
collIter = 10  # number of iterations of collision solving
collEps = 1e-5 # range within which to detect collisions
c_d = 1.0      # overlap repair coefficient  


@ti.kernel
def total_energy() -> ti.f32:

    E_k = 0.0
    for i in v:
        E_k += 0.5 * m * v[i].dot(v[i])
    return E_k


@ti.kernel
def advance_vel():

    for i in v:
        v[i] += h * g


# list of collisions
@ti.dataclass
class Collision:
    i : ti.i32  # index of first particle
    j : ti.i32  # index of first particle
    n : vec2    # unit normal pointing from j to i
    桑 : ti.f32  # magnitude of collision impulse
    d : ti.f32  # overlap depth

collisions = Collision.field(shape=(N*N+4*N,))
nC = ti.field(shape=(), dtype=ti.i32)

# list of (point, normal) pairs for boundary planes
boundaries = [
    [vec2(0,0), vec2(1,0)],
    [vec2(0,0), vec2(0,1)],
    [vec2(1,1), vec2(-1,0)],
    [vec2(1,1), vec2(0,-1)]
]

@ti.kernel
def cd_brute() -> ti.i32:
    nC[None] = 0

    # Detect and record object-wall collisions
    for i in range(N):
        for p, n in ti.static(boundaries):
            d = r - n.dot(x[i] - p)             
            if d > -collEps:
                k = ti.atomic_add(nC[None], 1)
                collisions[k].i = i
                collisions[k].j = -1  # -1 indicates a wall
                collisions[k].n = n
                collisions[k].桑 = 0
                collisions[k].d = d

    # Detect and record object-object collisions
    for i in range(N):
        for j in range(i):
            d = 2*r - tm.length(x[i] - x[j])
            if d > -collEps:
                k = ti.atomic_add(nC[None], 1)
                collisions[k].i = i
                collisions[k].j = j
                collisions[k].n = (x[i] - x[j]).normalized()
                collisions[k].桑 = 0
                collisions[k].d = d

    return nC[None]


桑_x = ti.Vector.field(2, shape=(N,), dtype=ti.f32)

@ti.kernel
def cr_impulse_GS_iter():

    # Compute sum of impulses affecting each object
    桑_x.fill(vec2(0))
    for k in range(nC[None]):
        C = collisions[k]
        if C.i >= 0: 桑_x[C.i] += C.桑 * C.n
        if C.j >= 0: 桑_x[C.j] -= C.桑 * C.n

    # Compute updates to all impulses in the context of others
    ti.loop_config(serialize=True)
    for k in range(nC[None]):
        C = collisions[k]
        i, j, n = C.i, C.j, C.n
        # get inverse mass and velocity appropriate to object or wall
        w_i, w_j, v_i, v_j = 0.0, 0.0, vec2(0.0), vec2(0.0)
        if i >= 0:
            w_i = 1/m
            v_i = v[i]
        if j >= 0:
            w_j = 1/m
            v_j = v[j]
        # compute impulse update        
        v_n = (v_i - v_j).dot(n)
        vel = -(1 + c_r) * v_n + c_d * C.d
        if i >= 0:
            vel -= n.dot(桑_x[i]) * w_i
        if j >= 0:
            vel += n.dot(桑_x[j]) * w_j
        m_eff = 1/(w_i + w_j)
        螖桑 = m_eff * vel
        # update impulse, and apply update also to impulse sums
        桑 = tm.max(0.0, C.桑 + 螖桑)
        螖桑 = 桑 - collisions[k].桑
        if i >= 0: 桑_x[i] += 螖桑 * n # delta gamma
        if j >= 0: 桑_x[j] -= 螖桑 * n
        # record new value for this collision impulse
        collisions[k].桑 = 桑 # gamma

@ti.kernel
def apply_impulses(debug : ti.i32):

    for k in range(nC[None]):
        C = collisions[k]
        m_i, m_j = m, m
        impulse = C.桑 * C.n
        if C.i >= 0: v[C.i] += impulse / m_i
        if C.j >= 0: v[C.j] -= impulse / m_j

    if debug:
        for k in range(nC[None]):
            C = collisions[k]
            print('collision', k, C.i, C.j, C.n, C.d, C.桑)
            v_i, v_j = vec2(0.0), vec2(0.0)
            if C.i > 0: v_i = v[C.i]
            if C.j > 0: v_j = v[C.j]
            print('  post: vel', v_i - v_j, 'vdotn', C.n.dot(v_i - v_j))


@ti.kernel
def advance_pos(t : ti.f32):
    for k in x:
        x[k] = x[k] + t * v[k]

# for visualizing contact graph
segInds = ti.field(shape=(8192,), dtype=ti.i32)

@ti.kernel
def fill_seginds():
    segInds.fill(0)
    for k in range(tm.min(4096,nC[None])):
        C = collisions[k]
        if C.i >= 0 and C.j >= 0:
            segInds[2*k+0] = C.i
            segInds[2*k+1] = C.j


# Create Taichi UI
window = ti.ui.Window("Collisions!", (width, width), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 0.5))

db = False
show_contacts = False
while window.running:

    advance_vel()
    nc = cd_brute()
    for i in range(collIter):
        cr_impulse_GS_iter()
    apply_impulses(db and nc > 5)
    advance_pos(h)

    if nc > 5:
        db = False

    # radius needs to be doubled on high dpi displays
    canvas.circles(x, radius=0.2*r, color=(0, 0, 0))

    if show_contacts:
        fill_seginds()
        canvas.lines(x, indices=segInds, width=0.0002, color=(0.8, 0.8, 0.8))

    window.show()

