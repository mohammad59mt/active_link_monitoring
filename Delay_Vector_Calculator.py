from __future__ import division
import random

# required global variables related to mathematical formulation
aggregated_equations, map_geneIndex_to_variableName, map_variableName_to_MAC = "",[], {}
class Required_function:
    @staticmethod
    def remove_hosts(node_based_path_array):
        res = [[] for _ in range(len(node_based_path_array))]
        path_len = len(node_based_path_array[0])
        for i in range(len(node_based_path_array)):
            path_len = len(node_based_path_array[i])
            res[i] = (node_based_path_array[i])[1:path_len-1]
        return res
    @staticmethod
    def routingMatrix_from_nodeBasePath(node_based_path_array):
        map_switch_to_MAC, map_MAC_to_switch, number_of_switches = {}, {}, 0
        number_of_probes = len(node_based_path_array)
        ''' find the all switches'''
        for path in node_based_path_array:
            for i in range(len(path)):
                if not path[i] in map_MAC_to_switch:
                    map_MAC_to_switch[path[i]] = number_of_switches
                    map_switch_to_MAC[number_of_switches] = path[i]
                    number_of_switches += 1
        routing_matrix =  [[[0 for f in range(number_of_probes)] for j in range(number_of_switches)] for i in range(number_of_switches)]
        tmp_f = 0
        for path in node_based_path_array:
            for i in range(len(path)-1):
                i, j = map_MAC_to_switch[path[i]], map_MAC_to_switch[path[i+1]]
                routing_matrix[i][j][tmp_f] = 1
            tmp_f += 1
        return routing_matrix, map_switch_to_MAC
    @staticmethod
    def generate_aggregated_equations(array_of_routing_matrices, array_of_delays, map_switch_to_MAC):
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
    @staticmethod
    def initial_population_float(min, max, pop_size):
        number_of_genes=len(map_geneIndex_to_variableName)
        two_floating_point = lambda x: float(int(x*100)/100.0)
        chrom = lambda :[two_floating_point(random.random()+random.randint(int(min), int(max)))+(min-int(min)) for i in range(number_of_genes)]
        pop = [chrom() for i in range(pop_size)]
        return pop
    @staticmethod
    def make_linear_equations(number_of_probes):
        number_of_unkowns = len(map_geneIndex_to_variableName)
        number_of_equations = number_of_probes
        b = [0 for _ in range(number_of_equations)]
        # a = [[0 for _ in range(number_of_equations)] for _ in range(number_of_unkowns)]
        a = [[0 for _ in range(number_of_unkowns)] for _ in range(number_of_equations)]
        equation_index = 0
        for equation in aggregated_equations.split('abs')[1:]:
            equation = equation.replace('(','').replace(')+','').replace(')','')
            for unknown in [eq.split('-')[0] for eq in equation.split('+')]:
                a[equation_index][map_geneIndex_to_variableName.index(unknown)] = 1
            b[equation_index] = float(equation.split('-')[1])
            equation_index += 1
        return a,b
    @staticmethod
    def compare_resutls(real_link_delay, measured_link_delay):
        res = {}
        max_difference = -1
        for link in real_link_delay:
            if link in measured_link_delay:
                res[link] = (real_link_delay[link], measured_link_delay[link])
                if max_difference < abs(real_link_delay[link] - measured_link_delay[link]):
                    max_difference = abs(real_link_delay[link] - measured_link_delay[link])
        return res, max_difference
    @staticmethod
    def numpy_suggestion_usingLSTSQ(node_based_path_array):
        a,b = Required_function.make_linear_equations(len(node_based_path_array))
        import numpy as np
        a, b = np.array(a), np.array(b)
        res = np.linalg.lstsq(a, b, rcond=None)
        opt_sol = res[0]
        return opt_sol
    #--- COST FUNCTION ------------------------------------------------------------+
    # function we are attempting to optimize (minimize)
    def fitness_function(x):
        tmp_aggregated_equations = aggregated_equations
        for i in range(len(x)-1, -1, -1):
            tmp_aggregated_equations = tmp_aggregated_equations.replace(map_geneIndex_to_variableName[i], str(x[i]))
            ''' genes must be positive'''
            if x[i] < 0: raise Exception('Negative value in a gene is not acceptable.')
        res = eval(tmp_aggregated_equations)
        return res
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
    def optimization(self,costFunc, bounds, max_itr, debug=False, optimal_fitness=None, acceptable_error=1e-5):
        history_of_best_fitnesses = [None,None,None]
        for i in range(max_itr):
            self.__one_iteration(costFunc,bounds)
            ''' keep the history of the best fitness, in case it didn't change for a while, mutate the population'''
            history_of_best_fitnesses = [history_of_best_fitnesses[1], history_of_best_fitnesses[2], int(self.fitness_best_g*100)]
            ''' if debug mode is active, show the best fitness of each iteration'''
            if debug: print(f'iter: {i:>4d}, best solution: {self.fitness_best_g:4.2f}')
            ''' if optimal_fitness is specified and we found that value, then end the search'''
            if not optimal_fitness is None and self.fitness_best_g-optimal_fitness<acceptable_error: return self.pos_best_g
            ''' if the fitness of the best particle doesn't change for a while, mutate 1_out_of_3 of the population'''
            if history_of_best_fitnesses[0]== history_of_best_fitnesses[1] == history_of_best_fitnesses[2]:
                for i in range(0,3,num_dimensions):
                    self.swarm[i].mutation(bounds,self.fitness_best_g)
                    if debug: print('mutated')
        return self.pos_best_g
    def measured_links_delay(self):
        measured_link_delay = {}
        tmp_indx = 0
        best_solution = self.pos_best_g
        make_tuple = lambda x: (x.split(', ')[0], x.split(', ')[1])
        for elem in map_geneIndex_to_variableName:
            measured_link_delay[make_tuple(map_variableName_to_MAC[elem])] = best_solution[tmp_indx]
            tmp_indx+=1
        return measured_link_delay
    def print_error(self,decimal_size=2):
        print(f'\nOptimization error: {self.fitness_best_g:4.{str(decimal_size)}f}')

def compare_results_and_print_output(real_link_delays, measured_link_delays):
    """ Compares the measured delays versus the real delays. To verify the quality of the algorithm.
    It cannot be used for real world scenarios (because in real-world we don't have the real-delay.
    :param real_link_delays: actual delay of each link
    :param measured_link_delays: measured delay for each link. Output of DVC
    :return: returns and prints comparison_of_real_and_measured_delay, max_error_per_link and summation_of_all_links
    """
    comparison, max_difference = Required_function.compare_resutls(real_link_delay=real_link_delays, measured_link_delay=measured_link_delays)
    link_delay_error = 0
    print('\n(source node, destination node): real delay --> measured delay')
    for link in comparison:
        print(link,':','{0:2.2f}'.format(comparison[link][0]),'-->','{0:2.2f}'.format(comparison[link][1]))
        link_delay_error += abs(comparison[link][0]-comparison[link][1])
    print("Max error per one link: ", max_difference)
    print('Summation of all link delays error: ', link_delay_error)
    return comparison, max_difference, link_delay_error
def delay_vector_calculator(array_of_delays, node_based_path_array, init_pop_min=1, init_pop_max =10, min=1, max=100, pop_size = 100,max_itereration= 150, debug=False):
    # if pop_size < 50: print("Population must be more than 50"); return
    random.seed(400)
    node_based_path_array = Required_function.remove_hosts(node_based_path_array)

    array_of_routing_matrices, map_switch_to_MAC = Required_function.routingMatrix_from_nodeBasePath(node_based_path_array)
    Required_function.generate_aggregated_equations(array_of_routing_matrices,array_of_delays, map_switch_to_MAC)

    if debug:
        print("Variables: ", map_geneIndex_to_variableName)
        print("Aggregated_equations: ", aggregated_equations)

    initial = Required_function.initial_population_float(init_pop_min, init_pop_max, pop_size)      # initial starting location [x1,x2...]

    opt_sol = Required_function.numpy_suggestion_usingLSTSQ(node_based_path_array)
    initial[0] = opt_sol

    bounds = [(min, max) for _ in range(len(initial[0]))]     # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    pso = PSO(initial)
    best_solution = pso.optimization(Required_function.fitness_function, bounds, max_itr=max_itereration, debug=debug, optimal_fitness=0.0, acceptable_error=1e-8)

    if debug:
        print('Optimizer solution:')
        print(best_solution)

    measured_link_delay = pso.measured_links_delay()

    print('\n(source node, destination node): measured delay')
    for link in measured_link_delay:
        print(link,':','{0:2.2f}'.format(measured_link_delay[link]))

    return measured_link_delay
# having two names for the same function
DVC = delay_vector_calculator

def sample_run():
    """ Read this sample to understand how to use this function"""
    ''' Sample inputs
        min: min value of each Gen -> min possible delay for a link
        init_pop_min: min value of each gen for initial population'''
    real_link_delays = {('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02'): 4, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01'): 4, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:04'): 3, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:02'): 3, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:12'): 6, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:01'): 6, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:05'): 3, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:12'): 3, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:04'): 9, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:05'): 9, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:03'): 9, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:04'): 9, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:06'): 9, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:04'): 9, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:07'): 8, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:06'): 8, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:08'): 5, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:07'): 5, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:09'): 3, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:08'): 3, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a'): 9, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:09'): 9, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0b'): 2, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0a'): 2, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0c'): 8, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0b'): 8, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0e'): 8, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0c'): 8, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0f'): 2, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0e'): 2, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:10'): 9, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0f'): 9, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:11'): 6, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:10'): 6, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:12'): 5, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:10'): 5, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:12'): 3, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:03'): 3, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:12'): 7, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:04'): 7, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:12'): 2, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:11'): 2, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0c'): 2, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:10'): 2, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:12'): 2, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0c'): 2, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0d'): 10, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:11'): 10, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0a'): 2, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0d'): 2, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:12'): 8, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:09'): 8, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:07'): 5, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:09'): 5, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:04'): 8, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:07'): 8, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0a'): 2, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:12'): 2}
    array_of_delays = [16.1, 16.1, 10.1, 10.1, 6.17, 20.1, 4.19, 12.1, 8.16, 4.17, 16.1, 4.19, 16.1, 4.16, 18.1, 6.17, 4.16, 18.1, 16.1]
    node_based_path_array = [['10.0.0.8', '00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:07', '10.0.0.8'], ['10.0.0.8', '00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:07', '10.0.0.8'], ['10.0.0.8', '00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:07', '10.0.0.8'], ['10.0.0.8', '00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:07', '10.0.0.8'], ['10.0.0.2', '00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:08', '10.0.0.2'], ['10.0.0.6', '00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0d', '10.0.0.6'], ['10.0.0.6', '00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0d', '10.0.0.6'], ['10.0.0.1', '00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:01', '10.0.0.1'], ['10.0.0.1', '00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', '10.0.0.1'], ['10.0.0.5', '00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0c', '10.0.0.5'], ['10.0.0.5', '00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0c', '10.0.0.5'], ['10.0.0.5', '00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0c', '10.0.0.5'], ['10.0.0.5', '00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0c', '10.0.0.5'], ['10.0.0.4', '00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0b', '10.0.0.4'], ['10.0.0.7', '00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:05', '10.0.0.7'], ['10.0.0.7', '00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:05', '10.0.0.7'], ['10.0.0.3', '00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0a', '10.0.0.3'], ['10.0.0.3', '00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a', '10.0.0.3'], ['10.0.0.9', '00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:09', '10.0.0.9']]
    ''' How to use the function'''
    measured_link_delays = DVC(array_of_delays, node_based_path_array, init_pop_min=1, init_pop_max =10, min=1, max=100, pop_size = 100,max_itereration= 150, debug=False)
    ''' How to check the accuracy if we have real-link-delays'''
    compare_results_and_print_output(real_link_delays=real_link_delays, measured_link_delays=measured_link_delays)
