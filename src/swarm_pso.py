import random as rd
import numpy as np
import math

INFINITY = 1 << 127

class PositionFactory:
    @staticmethod
    def generate_random(num_dimensions):
        return np.random.randn(num_dimensions, 2)

class Position:
    def __init__(self, num_dimensions):
        if num_dimensions != None:
            self.val = PositionFactory.generate_random(num_dimensions)

    def __repr__(self):
        return self.val.__str__()

    def get_val(self):
        return self.val
    
    def clone(self):
        result = Position(None)
        result.val = self.val.copy()
        return result

    def diff(self, other):
        assert(isinstance(other, Position))
        return self.val - other.get_val()
    
    def update(self, velocity, lr):
        self.val += velocity * lr

    def cost(self):
        return mccormick(self.val[0])

class Particle:
    def __init__(self, num_dimensions):
        self.pos = Position(num_dimensions)
        self.bpos = self.pos.clone()
        self.bval = self.val = self.pos.cost()
        self.velocity = np.zeros_like(self.pos.get_val())

    def update_velocity(self, idim, params, rp, rg, diff, gdiff):
        omega = params.omega
        phip = params.phip
        phig = params.phig

        self.velocity[idim] = omega * self.velocity[idim] + phip * rp * diff + phig * rg * gdiff

    def update(self, lr):
        self.pos.update(self.velocity, lr)
        self.val = self.pos.cost()

        if self.bval < self.val:
            self.bval = self.val
            self.bpos = self.pos.clone()

    def get_best_pos(self):
        return self.bpos

    def get_best_val(self):
        return self.bval

    def get_pos(self):
        return self.pos

    def get_val(self):
        return self.val

class Swarm:
    def __init__(self, parameters, num_particles, num_dimensions):
        self.swarm = []
        self.gbpos = None
        self.gbval = INFINITY
        self.num_dim = num_dimensions
        self.params = parameters

        for _ in range(num_particles):
            particle = Particle(num_dimensions)
            self.swarm.append(particle)

            pval = particle.get_best_val()
            if pval < self.gbval: # update global best pos & global best val
                self.gbpos = particle.get_best_pos().clone()
                self.gbval = pval

    def __repr__(self):
        return "Global best pos: {0}\nGlobal best val: {1}".format(self.gbpos, self.gbval)

    def run(self):
        self.iter()

        while not self.is_terminate():
            for particle in self.swarm:
                gdiff = self.gbpos.diff(particle.get_pos())
                diff = particle.get_best_pos().diff(particle.get_pos())
                rg = rd.random()

                for idim in range(self.num_dim):
                    rp = rd.random()
                    particle.update_velocity(idim, self.params, rp, rg, diff[idim], gdiff[idim])
                
                particle.update(lr=1.0)
                pval = particle.get_best_val()
                if pval < self.gbval:
                    self.gbpos = particle.get_best_pos().clone()
                    self.gbval = pval

            self.next()

    def is_terminate(self):
        return self.cur_gen > self.max_gen
    
    def iter(self):
        self.cur_gen = 1
        self.max_gen = 40

    def next(self):
        self.cur_gen += 1

def mccormick(x):
    (a, b) = x
    return math.sin(a + b) + (a - b) * (a - b) + 1.0 + 2.5 * b - 1.5 * a
 
class Parameters:
    def __init__(self, omega, phip, phig):
        self.omega = omega
        self.phip = phip
        self.phig = phig

def main():
    swarm = Swarm(Parameters(0.0, 0.6, 0.3), 100, 1)
    swarm.run()
    print(swarm)

main()
print("f(-.54719, -1.54719) : %.16f" % (mccormick([-.54719, -1.54719])))