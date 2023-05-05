import taichi as ti
import taichi.math as tm
vec2 = tm.vec2
vec3 = tm.vec3

mat3 = tm.mat3
from scene import *

# TODO expand this contact dataclass on your own
# @ti.dataclass
# class Contact:
#     # You may add in other fields below for your collision response implementation

# TODO: you can add other data class if you find that helpful for your
# implementation. (e.g. BoxState in scene_state.py)

# Contact Solver
@ti.data_oriented
class CollisionResponse:
    def __init__(self, scene_state, Cr, β, μ):
        self.scene = scene_state
        self.Cr = Cr
        self.β = β
        self.μ = μ
        self.init_contact()

    # TODO: Initialize the other fields you need for implementing collision response
    def init_contact(self):
        self.num_contact = ti.field(shape=(), dtype=ti.i32)
        # Initialize other variables you need below:
        # you can initialize them in the form of self.xx = ti.field(xxx)
        self.num_box = self.scene.boxes.shape[0]
        # len(self.scene.boxes)
        
        # inverse mass matrix
        self.M_inv = ti.field(shape=(3*self.num_box, 3*self.num_box), dtype=ti.f32)
        for i in range(self.num_box):
            self.M_inv[3 * i, 3 * i] = 1.0 / self.scene.boxes[i].m
            self.M_inv[3 * i + 1, 3 * i + 1] = 1.0 / self.scene.boxes[i].m
            self.M_inv[3 * i + 2, 3 * i + 2] = 1.0 / self.scene.boxes[i].I
        
        # Jacobian matrix
        self.J = ti.field(shape=(50, 3*self.num_box), dtype=ti.f32)
        
        # maximum seperation distance
        self.d = ti.field(shape=(50,), dtype=ti.f32)
        
        # collision index 
        self.idx = ti.field(shape=(50, 2), dtype=ti.i32)
        
        # contact impulse
        self.γ = ti.field(shape=(50,), dtype=ti.f32)
        self.γ0 = ti.field(shape=(50,), dtype=ti.f32)

    # clear all of your contacts
    # This function is called before the collision detection procerss.
    @ti.func
    def clearContact(self):
        self.num_contact[None] = 0
        
        # you can clear other attributes below if you need to do so

    # TODO: Implement this function that is going be triggered whenever a collision is being detected
    @ti.func
    def addContact(self, p1: vec2, r1: vec2, r2: vec2, n1: vec2, i1: int, i2: int, sep:float, nc: int):
        """
        This function is being triggered after the
        :param p1: vec2, the mass center of the reference rigid body
        :param r1: vec2, the displacement from the reference rigid body mass center to the contact point
        :param r2: vec2, the displacement from the incident rigid body mass center to the contact point
        :param n1: vec2, the normal of the reference edge
        :param i1: int, the index of the reference rigid body. You may find info related to this rigid body in self.state.boxes[i1]
        :param i2: int, the index of the incident rigid body. You may find info related to this rigid body in self.state.boxes[i2]
        :param sep: float, the maximum seperation distance between two boxes
        :param nc: int, number of contact points in between body i1 and i2.
        :return: void
        """
        # Note that if i1 < 0, the reference rigid body could be a rigid line boundary. Then, p1 would be a point on the
        # line boundary, r1 would be vec2(0, 0) and n1 would be the normal of the line boundary
        
        self.J[self.num_contact[None], 3 * i1] = n1[0]
        self.J[self.num_contact[None], 3 * i1 + 1] = n1[1]
        self.J[self.num_contact[None], 3 * i1 + 2] = crossXY(r1, n1)
        
        self.J[self.num_contact[None], 3 * i2] = -n1[0]
        self.J[self.num_contact[None], 3 * i2 + 1] = -n1[1]
        self.J[self.num_contact[None], 3 * i2 + 2] = -crossXY(r2, n1)
        
        self.d[self.num_contact[None]] = sep
        
        self.idx[self.num_contact[None], 0] = i1
        self.idx[self.num_contact[None], 1] = i2
        
        ti.atomic_add(self.num_contact[None], 1)

    def A(self, i, j):
        sum = 0
        for k in range(3 * self.num_box):
            sum += self.J[i, k] * self.M_inv[k, k] * self.J[j, k]
        return sum
        # return self.M_inv[i, j] * sum((self.J[i,k] * self.J[j,k] for k in range(self.num_box)))

    def b(self, V, i):
        b1 = 0 
        for k in range(3 * self.num_box):
            b1 += self.J[i, k] * V[k]
        b1 = - (1 + self.Cr) * b1
        b2 =  self.β * self.d[i]
        return b1 + b2

    # TODO: implemnt your projected Gauss-Seidel Contact solver below
    def PGS(self):
        num_box = self.scene.boxes.shape[0]
        
        # velocity vector
        V = ti.field(shape=(3*num_box, ), dtype=ti.f32)
        for i in range(num_box):
            V[3 * i] = self.scene.boxes[i].v[0]
            V[3 * i + 1] = self.scene.boxes[i].v[1]
            V[3 * i + 2] = self.scene.boxes[i].ω

        self.γ.fill(0)
        
        # cnt = 0
        Δγ = 1.0
        while ti.abs(Δγ) > 1e-4:
            Δγ = 0.0
            # print(cnt)
            for i in range(self.num_contact[None]):
                sum = 0
                for j in range(self.num_contact[None]):
                    sum += self.A(i, j) * self.γ[j]
                sum -= self.A(i, i) * self.γ[i]
                
                # print(sum)
                
                Aii = self.A(i, i)
                if Aii == 0:
                    val = 0
                else:
                    val = (self.b(V, i) - sum) / self.A(i, i)
                if val > 0:
                    val = 0
                
                # print(self.γ[i], val, val - self.γ[i])
                Δγ += (val - self.γ[i]) ** 2
                self.γ[i] = val
                # print(val)
                
                # project γ to [0, ∞)
                # if self.γ[i] < 0:
                #     self.γ[i] = 0

            Δγ = ti.sqrt(Δγ)
            # cnt += 1
            # print(Δγ)
        
        # print(self.γ)


    # TODO: update the velocities stored in self.state.boxes based on the impulses you solved for
    @ti.kernel
    def apply_impulses(self):
        for i in range(self.num_contact[None]):
            i1 = self.idx[i, 0]
            i2 = self.idx[i, 1]
            # self.scene.boxes[i1].v[0] += self.M_inv[3 * i1, 3 * i1] * self.γ[i] 
            # self.scene.boxes[i1].v[1] += self.M_inv[3 * i1 + 1, 3 * i1 + 1] * self.γ[i] 
            # self.scene.boxes[i1].ω += self.M_inv[3 * i1 + 2, 3 * i1 + 2] * self.γ[i]
            # self.scene.boxes[i2].v[0] -= self.M_inv[3 * i2, 3 * i2] * self.γ[i] 
            # self.scene.boxes[i2].v[1] -= self.M_inv[3 * i2 + 1, 3 * i2 + 1] * self.γ[i] 
            # self.scene.boxes[i2].ω -= self.M_inv[3 * i2 + 2, 3 * i2 + 2] * self.γ[i]
            self.scene.boxes[i1].v[0] += self.M_inv[3 * i1, 3 * i1] * self.γ[i] * self.J[i, 3 * i1]
            self.scene.boxes[i1].v[1] += self.M_inv[3 * i1 + 1, 3 * i1 + 1] * self.γ[i] * self.J[i, 3 * i1 + 1]
            self.scene.boxes[i1].ω += self.M_inv[3 * i1 + 2, 3 * i1 + 2] * self.γ[i] * self.J[i, 3 * i1 + 2]
            self.scene.boxes[i2].v[0] += self.M_inv[3 * i2, 3 * i2] * self.γ[i] * self.J[i, 3 * i2]
            self.scene.boxes[i2].v[1] += self.M_inv[3 * i2 + 1, 3 * i2 + 1] * self.γ[i] * self.J[i, 3 * i2 + 1]
            self.scene.boxes[i2].ω += self.M_inv[3 * i2 + 2, 3 * i2 + 2] * self.γ[i] * self.J[i, 3 * i2 + 2]
        
    #    for i in range(self.num_box):
    #         for j in range(self.num_contact[None]):
    #             self.scene.boxes[i].v[0] += self.M_inv[3 * i, 3 * i] * self.γ[3 * i] 
    #             self.scene.boxes[i].v[1] += self.M_inv[3 * i + 1, 3 * i + 1] * self.γ[3 * i + 1]
    #             self.scene.boxes[i].ω += self.M_inv[3 * i + 2, 3 * i + 2] * self.γ[3 * i + 2]






