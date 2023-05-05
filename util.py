import taichi as ti
import taichi.math as tm

vec2 = tm.vec2

@ti.func
def crossXY(u : vec2, v : vec2):
    """cross product of two xy plane vectors"""
    return u.x * v.y - u.y * v.x

@ti.func
def cross(w : float, v : vec2):
    """cross product of z-axis vector with xy-plane vector"""
    return vec2(-w*v.y, w*v.x)

@ti.func
def rot(q, X)->vec2:
    """rotate point X by rotation q"""
    return vec2(q.x * X.x - q.y * X.y, q.y * X.x + q.x * X.y)

@ti.func
def roti(q, X)->vec2:
    """rotate point X by the inverse of rotation q"""
    return vec2(q.x * X.x + q.y * X.y, -q.y * X.x + q.x * X.y)

@ti.func
def b2w(p, q, X):
    """body to world"""
    return p + rot(q, X)

@ti.func
def w2b(p, q, x):
    """world to body"""
    return roti(q, x - p)