from __future__ import division
import random
import math

aggregated_equations, map_geneIndex_to_variableName = "",{}
def generate_aggregated_equations(array_of_routing_matrices, array_of_delays):
    """ eq1 = a1, eq2 = a2 --->  abs(eq1-a1)+abs(eq2-a2)"""
    global aggregated_equations, map_geneIndex_to_variableName
    aggregated_equations = ''
    map_geneIndex_to_variableName = []
    number_of_probes, number_of_switches = len(array_of_routing_matrices[0][0]), len(array_of_routing_matrices[0])
    for probe in range(number_of_probes):
        aggregated_equations += 'abs('
        for i in range(number_of_switches):
            for j in range(number_of_switches):
                if array_of_routing_matrices[i][j][probe]:
                    aggregated_equations += ('x'+str(i*number_of_switches+j)+'+')
                    if ('x'+str(i*number_of_switches+j)) not in map_geneIndex_to_variableName:
                        map_geneIndex_to_variableName.append('x'+str(i*number_of_switches+j))
        aggregated_equations += '0-'+str(array_of_delays[probe])+')+'
    aggregated_equations += '0'
    aggregated_equations, map_geneIndex_to_variableName = aggregated_equations.replace('+0',''), sorted(map_geneIndex_to_variableName)
def generate_aggregated_equations_NEW(array_of_routing_matrices, array_of_delays, map_switch_to_MAC):
    """ eq1 = a1, eq2 = a2 --->  abs(eq1-a1)+abs(eq2-a2)"""
    global aggregated_equations, map_geneIndex_to_variableName, map_variableName_to_MAC
    aggregated_equations = ''
    map_variableName_to_MAC = {}
    map_geneIndex_to_variableName = []
    number_of_probes, number_of_switches = len(array_of_routing_matrices[0][0]), len(array_of_routing_matrices[0])
    for probe in range(number_of_probes):
        aggregated_equations += 'abs('
        for i in range(number_of_switches):
            for j in range(number_of_switches):
                if array_of_routing_matrices[i][j][probe]:
                    aggregated_equations += ('x'+str(i*number_of_switches+j)+'+')
                    if ('x'+str(i*number_of_switches+j)) not in map_geneIndex_to_variableName:
                        map_geneIndex_to_variableName.append('x'+str(i*number_of_switches+j))
                        map_variableName_to_MAC['x'+str(i*number_of_switches+j)] = str(map_switch_to_MAC[i])+', '+str(map_switch_to_MAC[j])
        aggregated_equations += '0-'+str(array_of_delays[probe])+')+'
    aggregated_equations += '0'
    aggregated_equations, map_geneIndex_to_variableName = aggregated_equations.replace('+0',''), sorted(map_geneIndex_to_variableName)
def initial_population_float(min, max, pop_size):
    number_of_genes=len(map_geneIndex_to_variableName)
    two_floating_point = lambda x: float(int(x*100)/100.0)
    chrom = lambda :[two_floating_point(random.random()+random.randint(int(min), int(max)))+(min-int(min)) for i in range(number_of_genes)]
    pop = [chrom() for i in range(pop_size)]
    return pop
#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    tmp_aggregated_equations = aggregated_equations
    for i in range(len(x)-1, -1, -1):
        tmp_aggregated_equations = tmp_aggregated_equations.replace(map_geneIndex_to_variableName[i], str(x[i]))
        ''' genes must be positive'''
        if x[i] < 0: raise Exception('Negative value in a gene is not acceptable.')
    res = eval(tmp_aggregated_equations)
    return res

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self, initial_value):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.best_fitness=-1          # best fitness individual
        self.fitness=-1               # fitness individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(initial_value[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.fitness=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.fitness<self.best_fitness or self.best_fitness==-1:
            self.pos_best_i=self.position_i.copy()
            self.best_fitness=self.fitness

    # update new particle velocity
    def update_velocity(self,pos_best_g, velocity=None):
        if velocity is None:
            w=0.5       # constant inertia weight (how much to weigh the previous velocity)  default 0.5
            c1=1        # cognative constant (attraction to best particle vs selfishness)   default 1
            c2=2        # social constant   (attraction to social behaviour)   default 2
        else:
            w, c1, c2 = velocity

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]
                self.velocity_i[i] *= -1

            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                self.velocity_i[i] *= -1

    # prevent from getting stuck in local optimume
    def mutation(self, bounds, best_fitness):
        if self.fitness != best_fitness:
            self.position_i=[]          # particle position
            self.velocity_i=[]          # particle velocity
            self.pos_best_i=[]          # best position individual
            self.best_fitness=-1          # best fitness individual
            self.fitness=-1               # fitness individual

            two_floating_point = lambda x: float(int(x*100)/100.0)
            initial_value = [two_floating_point(random.random()+random.randint(int(bounds[i][0]), int(bounds[i][1])))+(bounds[i][0]-int(bounds[i][0])) for i in range(num_dimensions)]

            for i in range(0, num_dimensions):
                self.velocity_i.append(random.uniform(-1,1))
                self.position_i.append(initial_value[i])

class PSO():
    def old_initial(self, costFunc, initial_population, bounds, maxiter, verbose=False):
        global num_dimensions

        num_particles = len(initial_population)
        num_dimensions = len(initial_population[0])
        fitness_best_g=-1                   # best fitness for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        # for i in range(0,num_particles):
        #     swarm.append(Particle(x0))
        for x0 in initial_population:
            swarm.append(Particle(x0))
        # begin optimization loop
        i=0
        while i<maxiter:
            if verbose: print(f'iter: {i:>4d}, best solution: {fitness_best_g:10.6f}')
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].fitness<fitness_best_g or fitness_best_g==-1:
                    pos_best_g=list(swarm[j].position_i)
                    fitness_best_g=float(swarm[j].fitness)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        print('\nFINAL SOLUTION:')
        print(f'   > {pos_best_g}')
        print(f'   > {fitness_best_g}\n')
    def __init__(self, initial_population):
        global num_dimensions
        num_dimensions = len(initial_population[0])
        self.fitness_best_g=-1               # best fitness for group
        self.pos_best_g=[]                   # best position for group

        # establish the swarm
        self.swarm=[]
        for x0 in initial_population:
            self.swarm.append(Particle(x0))
    def __one_iteration(self, costFunc, bounds):
        # cycle through particles in swarm and evaluate fitness
        for swarm in self.swarm:
            swarm.evaluate(costFunc)

            # determine if current particle is the best (globally)
            if swarm.fitness< self.fitness_best_g or self.fitness_best_g==-1:
                self.pos_best_g=list(swarm.position_i)
                self.fitness_best_g=float(swarm.fitness)

        # cycle through swarm and update velocities and position
        for swarm in self.swarm:
            swarm.update_velocity(self.pos_best_g)
            swarm.update_position(bounds)
    def optimization(self,costFunc, bounds, max_itr, debug=False):
        history_of_best_fitnesses = [-1,-1,-1]
        for i in range(max_itr):
            self.__one_iteration(costFunc,bounds)
            history_of_best_fitnesses = [history_of_best_fitnesses[1], history_of_best_fitnesses[2], int(self.fitness_best_g*100)]
            if debug: print(f'iter: {i:>4d}, best solution: {self.fitness_best_g:4.2f}')
            if history_of_best_fitnesses[0]== history_of_best_fitnesses[1] == history_of_best_fitnesses[2]:
                for i in range(0,3,num_dimensions):
                    self.swarm[i].mutation(bounds,self.fitness_best_g)
                    print('mutated')


    def print_best_result(self,decimal_size=2):
        # print final results
        print('\nFINAL SOLUTION:')
        best_pos = [int(element*pow(10,decimal_size))/float(pow(10,decimal_size)) for element in self.pos_best_g]
        # print(f'   > {self.pos_best_g}')
        print(f'   > {best_pos}')
        print(f'   > {self.fitness_best_g:4.{str(decimal_size)}f}\n')

if __name__ == '__main__':
    random.seed(400)
    array_of_routing_matrices = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    # array_of_delays = [18.822222222222223, 25.4, 31.27179487179487, 13.733333333333334, 33.64545454545455, 23.7, 47.33170731707317, 27.414285714285715, 15.1475, 21.88974358974359, 34.45128205128205, 21.4027027027027, 25.42439024390244, 19.016216216216215, 13.977777777777778, 21.676923076923078, 34.321739130434786, 25.546341463414635, 30.264864864864865, 21.966666666666665, 29.745238095238093, 18.3325, 20.33877551020408, 14.555, 17.961538461538463, 26.474999999999998, 23.18, 14.232432432432432, 18.50263157894737, 29.174358974358974, 40.876315789473686, 17.96842105263158, 15.087179487179487, 21.210526315789473, 26.38974358974359, 25.86829268292683, 17.297777777777778, 18.165, 29.76279069767442, 21.25526315789474, 25.638461538461538, 21.8875, 46.85945945945946]
    # array_of_delays = [14,22,14,10,30,22,42,22,10,18,30,18,22,14,10,18,30,22,14,18,26,14,14,10,14,22,18,10,14,26,30,14,10,18,22,22,14,14,14,18,22,18,42]
    array_of_delays = [10,10,12,14,34,36,22,32,48,12,14,16,18,26,20,40,58,24,32,40,60,22,58,68,70,48,40,74]
    map_switch_to_MAC = {0: '00:00:00:00:00:00:00:01', 1: '00:00:00:00:00:00:00:02', 2: '00:00:00:00:00:00:00:03', 3: '00:00:00:00:00:00:00:04', 4: '00:00:00:00:00:00:00:05', 5: '00:00:00:00:00:00:00:06', 6: '00:00:00:00:00:00:00:07', 7: '00:00:00:00:00:00:00:08', 8: '00:00:00:00:00:00:00:09', 9: '00:00:00:00:00:00:00:0a'}
    # array_of_routing_matrices, array_of_delays = [[[0,0,1],[1,0,0],[0,1,0]], [[0,1,0],[1,0,0],[0,0,0]], [[0,1,0],[0,0,1],[1,0,0]], [[0,0,1],[0,0,0],[1,0,0]]],   [5.38, 2.96, 5.38, 3.12]
    generate_aggregated_equations(array_of_routing_matrices,array_of_delays)
    generate_aggregated_equations_NEW(array_of_routing_matrices,array_of_delays, map_switch_to_MAC)
    print(map_geneIndex_to_variableName)

    init_pop_min, init_pop_max =0.1, 100
    min, max = 0.1, 100
    pop_size = 100
    max_itereration=300
    initial = initial_population_float(init_pop_min, init_pop_max, pop_size)      # initial starting location [x1,x2...]
    bounds = [(min, max) for i in range(len(initial))]     # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    pso = PSO(initial)
    pso.optimization(func1, bounds, max_itr=max_itereration, debug=True)
    pso.print_best_result(decimal_size=2)

    for elem in map_geneIndex_to_variableName:
        print(map_variableName_to_MAC[elem])


