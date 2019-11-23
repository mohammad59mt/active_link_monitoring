import numpy as np, datetime
from pulp import *

class backup_monitoring:
    """F: Number of flows,  N: Number of nodes, L: Maximum length of monitoring routes
        x: x monitoring flows will cross each link, src: Sources of each flow,  dst: Destination of each flow
        A: Link adjacency matrix (0, 1),   B: Capacity matrix,  mu: Monitoring traffic ratio to total traffic
        R: Routing Matrix  ---> BLP Variable"""
    def __init__(self, in_topo=None):
        self.L = 60  # Maximum length of monitoring routes
        self.x = 1  # x monitoring flows will cross each link
        self.mu = 0.01  # max ratio of monitoring traffic to link capacity
        self.__convert_topo_to_desigred_format(in_topo) # N, number_of_hosts, A, map_switch_to_MAC, possible_sources, map_sourceSwitch_to_host
        self.B = 10000000 * self.A  # links are considered to be 10Mb
        ''' flows adjustment'''
        self.F = self.number_of_links_between_switches*self.x  # Number of Flows
        temp = len(self.possible_sources)
        self.src = [self.possible_sources[i%temp] for i in range(self.F)]  # Sources of each flow
        self.dst = self.src  # Destination of each flow
        self.flow_rate = 1  # monitoring flows are considered to be 1Mbps
        self.__forwarding_table_entries = None
        self.monitoring_lp_problem = None
        self.routing_matrix = [[[0 for f in range(self.F)] for j in range(self.N)] for i in range(self.N)]
        self.big_number = 51*(self.N*self.N)-150*self.N+100
    def __convert_topo_to_desigred_format(self, topo):
        """ """
        if topo is None: topo = {('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 's'): 0, ('7a:1b:c3:65:e5:4a', '00:00:00:00:00:00:00:02', 'h'): 1, ('96:47:ee:7d:c4:3e', '00:00:00:00:00:00:00:01', 'h'): 1, ('ce:74:73:a5:e7:53', '00:00:00:00:00:00:00:03', 'h'): 1}
        '''Find the set of switches and hosts'''
        set_of_switchs, set_of_hosts = set(), set()
        def is_MAC(input):
            if len(input.split(':')) == 8: return True
            else: return False
        for element in topo:
            # each "element" is either (MAC, MAC, 's') OR (IP, MAC, 'h')
            for i in range(0,2):
                if is_MAC(element[i]): set_of_switchs.add(element[i])
                else: set_of_hosts.add(element[i])
        ''' Set parameters --> N: number of nodes (switches),  H: number of hosts, A: adjacency matrix'''
        number_of_links_between_switches = 0
        A = np.zeros((len(set_of_switchs), len(set_of_switchs)))
        # A = [[0 for j in range(len(set_of_switchs))] for i in range(len(set_of_switchs))]
        map_MAC_to_switch = {}
        tN, number_of_hosts = 0, 0  # tN: number of different IPs viewed till now
        possible_sources = set()  # s: set of switches with possibility to be a source (connected to a host)
        map_sourceSwitch_to_host = {}
        for element in topo:
            value = topo[element]
            ''' if both nodes are switch'''
            if element[2] == 's':
                # each "element" is either (MAC, MAC, 's') OR (IP, MAC, 'h')
                for i in range(0, 2):
                    if element[i] not in map_MAC_to_switch:
                        map_MAC_to_switch[element[i]] = tN
                        tN += 1
                if int(value) == 1:
                    # A[map_MAC_to_switch[element[0]], map_MAC_to_switch[element[1]]] = 1
                    A[map_MAC_to_switch[element[0]]][map_MAC_to_switch[element[1]]] = 1
                    number_of_links_between_switches += 1
            # ''' if one node is a switch and another one is a host '''
            elif element[2] == 'h':
                number_of_hosts += 1
                switch = element[1] if is_MAC(element[1]) else element[0]
                host = element[0] if is_MAC(element[1]) else element[1]
                ''' if the connected switch doesn't have a mapping number (from IP to switch number) then assign one'''
                if switch not in map_MAC_to_switch:
                    map_MAC_to_switch[switch] = tN
                    tN += 1
                ''' add the switch number to set of possible sources'''
                possible_sources.add(map_MAC_to_switch[switch])
                ''' map the host to to a source switch'''
                if switch in map_sourceSwitch_to_host:
                    map_sourceSwitch_to_host[switch].append(host)
                else:
                    map_sourceSwitch_to_host[switch] = [host]
        map_switch_to_MAC = {}
        for element in map_MAC_to_switch:
            map_switch_to_MAC[map_MAC_to_switch[element]] = element

        self.number_of_links_between_switches = number_of_links_between_switches
        self.N, self.number_of_hosts, self.A, self.map_switch_to_MAC, self.possible_sources, self.map_sourceSwitch_to_host = \
            tN, number_of_hosts, A, map_switch_to_MAC, list(possible_sources), map_sourceSwitch_to_host
    def __routing_matrix(self):
        """"""
        ''' to simplify the writing'''
        F, N, src, dst, solved_problem = self.F, self.N, self.src, self.dst, self.monitoring_lp_problem
        def index_extractor(input):
            """ remove waste characters """
            input = input.replace('(', '').replace(',', '').replace(')', '')
            return int(str(input).split('_')[1]), int(str(input).split('_')[2]), int(str(input).split('_')[3])
        ''' create route-matrix'''
        routing_matrix = [[[0 for f in range(F)] for j in range(N)] for i in range(N)]
        for variable in solved_problem.variables():
            if int(variable.varValue) is not 0:
                if 'R_(' in variable.name:
                    x, y, z = index_extractor(variable.name)
                    routing_matrix[x][y][z] = int(variable.varValue)
                    self.routing_matrix[x][y][z] = max(routing_matrix[x][y][z], self.routing_matrix[x][y][z])
        return routing_matrix
    def __convert_to_routing_rule_entries(self):
        """"""
        ''' to simplify the writing'''
        F, N, src, dst, solved_problem, map_sourceSwitch_to_host, map_switch_to_MAC = \
            self.F, self.N, self.src, self.dst, self.monitoring_lp_problem, self.map_sourceSwitch_to_host, self.map_switch_to_MAC

        routing_matrix = self.__routing_matrix()
        # def index_extractor(input):
        #     """ remove waste characters """
        #     input = input.replace('(', '').replace(',', '').replace(')', '')
        #     return int(str(input).split('_')[1]), int(str(input).split('_')[2]), int(str(input).split('_')[3])
        def find_next_hob(f, i):
            for j in range(N):
                if routing_matrix[i][j][f] is not 0:
                    return j
            ''' if there is not any next hob then return current hob'''
            return -1
        # ''' create route-matrix'''
        # routing_matrix = [[[0 for f in range(F)] for j in range(N)] for i in range(N)]
        # for variable in solved_problem.variables():
        #     if int(variable.varValue) is not 0:
        #         if 'R_(' in variable.name:
        #             x, y, z = index_extractor(variable.name)
        #             routing_matrix[x][y][z] = int(variable.varValue)

        ''' create routes-rule-entries'''
        routes_rule_entries = [[] for f in range(F)]
        for f in range(F):
            tmp_src = src[f]
            # add source host
            routes_rule_entries[f].append(map_sourceSwitch_to_host[map_switch_to_MAC[tmp_src]][0])
            # add switches in the route
            routes_rule_entries[f].append(map_switch_to_MAC[tmp_src])
            ''' In some cases the source and destination are the same (loop): find next hob '''
            tmp_src = find_next_hob(f, tmp_src)
            # if the next hob is not -1 (meaning there is not any next hob) add that to route-rules
            if tmp_src is not -1:
                routes_rule_entries[f].append(map_switch_to_MAC[tmp_src])
            while tmp_src is not dst[f] and tmp_src is not -1:
                ''' find next hob '''
                tmp_src = find_next_hob(f, tmp_src)
                # if next-hob is -1 means there is not any next hob
                if tmp_src is not -1:
                    routes_rule_entries[f].append(map_switch_to_MAC[tmp_src])
            #add destination host
            routes_rule_entries[f].append(map_sourceSwitch_to_host[map_switch_to_MAC[dst[f]]][0])
        if self.__forwarding_table_entries is None:
            self.__forwarding_table_entries = routes_rule_entries
        else:
            # merge new and existing forwarding table entries
            garbage_output = [self.__forwarding_table_entries.append(entry) for entry in routes_rule_entries]
    def __purge_redundant_flows(self):
        tempF = len(self.__forwarding_table_entries)
        for f in range(tempF-1, -1 , -1):
            ''' removing flows doesn't leave the source switch'''
            if len(self.__forwarding_table_entries[f]) == 3 or (len(self.__forwarding_table_entries[f]) == 4 and self.__forwarding_table_entries[f][1] == self.__forwarding_table_entries[f][2]):
                self.F -= 1
                del self.__forwarding_table_entries[f]
                for i in range(self.N):
                    for j in range(self.N):
                        del self.routing_matrix[i][j][f]
    def solve_optimal(self):
        """ Three steps: 1. Create Problem,  2. Define Variables, and 3. Define Constraints"""
        ''' Step 1. create a (I)LP problem '''
        monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpProblem)

        ''' to simplify the writing '''
        N, F, src, dst, A, mu, L, flow_rate, B, x = self.N, self.F, self.src, self.dst, self.A, self.mu, self.L, self.flow_rate, self.B, self.x

        '''Step 2. define (I)LP variables --> cat={'Binary', 'Continuous', 'Integer'} '''
        R = pulp.LpVariable.dicts('R', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), cat='Binary')
        P = pulp.LpVariable.dicts('P', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), lowBound=0, upBound=L + 1, cat='Integer')
        y = pulp.LpVariable('y',lowBound=0,upBound=1,cat='Binary')    #meta-variable used to make sure there is not any repeated path (similar path for two probes)

        '''Objective function'''
        monitoring_lp_problem += R[0, 0, 0]
        ''' -1 for MAX the objective function and 1 for minimization of that'''
        monitoring_lp_problem.sense = 1

        '''Step 3.  Define constraints '''
        '''****************************** DO NOT USE < OR > *****************************
        ************************ JUST DO USE >= OR <= OR == *****************************
        ************** NUMPY VALUES ARE DIFFERENT FROM SIMPLE VALUES ********************
        ******************** NUMPY.INT64(0) is not the same as 0 *********************'''
        ''' Don'nt use links that doesn't exist'''
        for i in range(N):
            for j in range(N):
                for f in range(F):
                    # monitoring_lp_problem += R[i, j, f] <= int(A[i, j])
                    monitoring_lp_problem += R[i, j, f] <= int(A[i][j])
        ''' From each link, at least x flows pass'''
        for i in range(N):
            for j in range(N):
                # if int(A[i, j]) is not 0:
                if int(A[i][j]) is not 0:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) >= x
        ''' Flow conservation for R'''
        for f in range(F):
            for i in range(N):
                if src[f] == dst[f] == i:
                    # in this case, loop is allowed
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
                elif i == src[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 1
                elif i == dst[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == -1
                else:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
        ''' Use at most miu percent of bandwidth for monitoring'''
        for i in range(N):
            for j in range(N):
                # monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) * flow_rate <= mu * float(B[i, j])
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) * flow_rate <= mu * float(B[i][j])
        ''' No loop (except for case src[f]==dst[f]): do not leave a node twice'''
        for f in range(F):
            for i in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) <= 1  # circle
        ''' Keep the length of monitoring routes less than a predefined value'''
        for f in range(F):
            monitoring_lp_problem += pulp.lpSum([R[i, j, f] for i in range(N) for j in range(N)]) <= L
        ''' If R[i, j, f] is zero then P[i, j, f] is zero'''
        for f in range(F):
            for i in range(N):
                for j in range(N):
                    monitoring_lp_problem += P[i, j, f] <= (L + 1) * R[i, j, f]
        ''' Add the value of P by one, after each step'''
        for f in range(F):
            for i in range(N):
                if not i == src[f] == dst[f]:
                    monitoring_lp_problem += lpSum([P[i, j, f] for j in range(N)]) == lpSum(
                        [P[j, i, f] + R[i, j, f] for j in range(N)])
        ''' Leave source node and enter destination node (to make sure there is a path for each flow, otherwise, some flows will not come out of source)'''
        for f in range(F):
            monitoring_lp_problem += lpSum([R[src[f], j, f] for j in range(N)]) == 1
            monitoring_lp_problem += lpSum([R[i, dst[f], f] for i in range(N)]) == 1
        ''' No repeated path (n-equation m-unknown where n<m has infinite answers, n>m has no answer, n==m has one answer'''
        for f in range(F):
            for f_prime in range(f+1, F):
                monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) <= y*self.big_number
                monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) >= -y*self.big_number
                monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) <= -1 + (1-y)*self.big_number
                monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) >= 1 -y*self.big_number
                # if src[f] == src[f_prime] and dst[f] == dst[f_prime]:
                #     monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])<=\
                #                              pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) - 1
        self.monitoring_lp_problem = monitoring_lp_problem

        # self.monitoring_lp_problem.setSolver(pulp.GLPK_CMD)
        # self.monitoring_lp_problem.solve(pulp.GLPK_CMD(msg=1, options=["--tmlim", "10"]))

        self.monitoring_lp_problem.solve()

        # self.monitoring_lp_problem.parameters.timelimit.set(300.0)

        # P = monitoring_lp_problem
        # Build the solverModel for your preferred
        # solver = pulp.GLPK_CMD()
        # solver.buildSolverModel(P)

        #Modify the solvermodel
        # solver.solverModel.parameters.timelimit.set(60)

        #Solve P
        # solver.callSolver(P)
        # status = solver.findSolutionValues(P)

    def solve_for_some_links(self, links_to_be_crossed, number_of_flows):
        """ Three steps: 1. Create Problem,  2. Define Variables, and 3. Define Constraints"""
        ''' Step 1. create a (I)LP problem '''
        monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpProblem)

        ''' to simplify the writing '''
        N, src, dst, A, mu, flow_rate, L, flow_rate, B, x = self.N, self.src, self.dst, self.A, self.mu, self.flow_rate, self.L, self.flow_rate, self.B, self.x
        F = number_of_flows

        '''Step 2. define (I)LP variables --> cat={'Binary', 'Continuous', 'Integer'} '''
        R = pulp.LpVariable.dicts('R', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), cat='Binary')
        P = pulp.LpVariable.dicts('P', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), lowBound=0, upBound=L + 1, cat='Integer')
        # P shows the step of each movement -- using this variable we prevent flows from making a isolated loop along with the main route from source to destination

        '''Objective function'''
        monitoring_lp_problem += R[0, 0, 0]
        ''' -1 for MAX the objective function and 1 for minimization of that'''
        monitoring_lp_problem.sense = 1

        '''Step 3.  Define constraints '''
        '''****************************** DO NOT USE < OR > *****************************
        ************************ JUST DO USE >= OR <= OR == *****************************'''
        ''' Don'nt use links that doesn't exist'''
        for i in range(N):
            for j in range(N):
                for f in range(F):
                    monitoring_lp_problem += R[i, j, f] <= int(A[i, j])
        ''' From each link, exactly x flows pass'''
        for i in range(N):
            for j in range(N):
                if int(A[i, j]) is not 0:
                    if links_to_be_crossed[i,j]:
                        monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) >= x
        ''' Flow conservation for R'''
        for f in range(F):
            for i in range(N):
                if src[f] == dst[f] == i:
                    # in this case, loop is allowed
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
                elif i == src[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 1
                elif i == dst[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == -1
                else:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
        ''' Use at most miu percent of bandwidth for monitoring'''
        for i in range(N):
            for j in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) * flow_rate <= mu * float(B[i, j])
        ''' No loop (except for case src[f]==dst[f]): do not leave a node twice'''
        for f in range(F):
            for i in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) <= 1  # circle
        ''' Keep the length of monitoring routes less than a predefined value'''
        for f in range(F):
            monitoring_lp_problem += pulp.lpSum([R[i, j, f] for i in range(N) for j in range(N)]) <= L
        ''' If R[i, j, f] is zero then P[i, j, f] is zero'''
        for f in range(F):
            for i in range(N):
                for j in range(N):
                    monitoring_lp_problem += P[i, j, f] <= (L + 1) * R[i, j, f]
        ''' Add the value of P by one, after each step'''
        for f in range(F):
            for i in range(N):
                if not i == src[f] == dst[f]:
                    monitoring_lp_problem += lpSum([P[i, j, f] for j in range(N)]) == lpSum(
                        [P[j, i, f] + R[i, j, f] for j in range(N)])
        self.monitoring_lp_problem = monitoring_lp_problem
        self.monitoring_lp_problem.solve()
    def solve_incremental(self):
        for i in range(self.x):
            centinel = 10
            # links_to_be_crossed = self.A.copy()
            links_to_be_crossed = [[int(self.A[i,j]) for j in range(self.N)] for i in range(self.N)]
            while centinel>0 and sum([sum(lines) for lines in links_to_be_crossed]):
                print('round {}'.format(11-centinel))
                centinel -= 1
                self.__solve_and_maximize_passed_links(links_to_be_crossed, number_of_flows=3)
                if self.optimality_status() is 'Optimal':
                    #remove links that are crossed already from links_to_be_crossed
                    routing_matrix = self.__routing_matrix()
                    links_to_be_crossed = [[int(links_to_be_crossed[i][j] and (not sum(routing_matrix[i][j]))) for j in range(self.N)] for i in range(self.N)]
                    #update routing rules
                    self.__convert_to_routing_rule_entries()
                    # self.__purge_redundant_flows()
                else: raise Exception('Problem is not solvable')
        self.__purge_redundant_flows()
    def __solve_and_maximize_passed_links(self, links_to_be_crossed, number_of_flows):
        """ Three steps: 1. Create Problem,  2. Define Variables, and 3. Define Constraints"""
        ''' Step 1. create a (I)LP problem '''
        monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpProblem)

        ''' to simplify the writing '''
        N, src, dst, A, mu, flow_rate, L, flow_rate, B, x = self.N, self.src, self.dst, self.A, self.mu, self.flow_rate, self.L, self.flow_rate, self.B, self.x
        F = number_of_flows

        '''Step 2. define (I)LP variables --> cat={'Binary', 'Continuous', 'Integer'} '''
        R = pulp.LpVariable.dicts('R', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), cat='Binary')
        P = pulp.LpVariable.dicts('P', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), lowBound=0, upBound=L + 1, cat='Integer')
        # P shows the step of each movement -- using this variable we prevent flows from making a isolated loop along with the main route from source to destination

        '''Objective function'''
        monitoring_lp_problem += pulp.lpSum([R[i, j, f]*links_to_be_crossed[i][j] for i in range(N) for j in range(N) for f in range(F)])
        ''' -1 for MAX the objective function and 1 for minimization of that'''
        monitoring_lp_problem.sense = -1

        '''Step 3.  Define constraints '''
        '''****************************** DO NOT USE < OR > *****************************
        ************************ JUST DO USE >= OR <= OR == *****************************'''
        ''' Don't use links that doesn't exist'''
        for i in range(N):
            for j in range(N):
                for f in range(F):
                    monitoring_lp_problem += R[i, j, f] <= int(A[i, j])
        ''' From each link, exactly x flows pass'''
        for i in range(N):
            for j in range(N):
                if int(A[i, j]) is not 0:
                    if links_to_be_crossed[i,j]:
                        monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) >= x
        ''' Flow conservation for R'''
        for f in range(F):
            for i in range(N):
                if src[f] == dst[f] == i:
                    # in this case, loop is allowed
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
                elif i == src[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 1
                elif i == dst[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == -1
                else:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
        ''' Use at most miu percent of bandwidth for monitoring'''
        for i in range(N):
            for j in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) * flow_rate <= mu * float(B[i, j])
        ''' No loop (except for case src[f]==dst[f]): do not leave a node twice'''
        # for f in range(F):
        #     for i in range(N):
        #         monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) <= 1  # circle
        ''' Keep the length of monitoring routes less than a predefined value (it is required for next constraint which is CNST2'''
        for f in range(F):
            monitoring_lp_problem += pulp.lpSum([R[i, j, f] for i in range(N) for j in range(N)]) <= L
        ''' CNST2: If R[i, j, f] is zero then P[i, j, f] is zero'''
        for f in range(F):
            for i in range(N):
                for j in range(N):
                    monitoring_lp_problem += P[i, j, f] <= (L + 1) * R[i, j, f]
        ''' Add the value of P by one, after each step'''
        for f in range(F):
            for i in range(N):
                if not i == src[f] == dst[f]:
                    monitoring_lp_problem += lpSum([P[i, j, f] for j in range(N)]) == lpSum(
                        [P[j, i, f] + R[i, j, f] for j in range(N)])
        self.monitoring_lp_problem = monitoring_lp_problem
        self.monitoring_lp_problem.solve()
    def forwarding_table_entries(self):
        if self.monitoring_lp_problem is None:
            raise Exception('Problem is not solved yet')
        if self.optimality_status() is 'Optimal' and self.__forwarding_table_entries is None:
            self.__convert_to_routing_rule_entries()
            self.__purge_redundant_flows()
        return self.__forwarding_table_entries
    def optimality_status(self):
        if monitoring is None: raise Exception('Problem is not defined yet')
        return LpStatus[self.monitoring_lp_problem.status]
    def print_results(self):
        if self.monitoring_lp_problem is not None:
            print('Problem Solving Status: {}'.format(pulp.LpStatus[self.monitoring_lp_problem.status]))
            print("Objective function value: {}".format(pulp.value(self.monitoring_lp_problem.objective)))
            print('Variables: ')
            for variable in self.monitoring_lp_problem.variables():
                if int(variable.varValue) is not 0:
                    print("   {} = {}".format(variable.name, variable.varValue))
        else:
            raise Exception('The problem is not solved yet.')

class monitoring:
    """F: Number of flows,  N: Number of nodes, L: Maximum length of monitoring routes
        x: x monitoring flows will cross each link, src: Sources of each flow,  dst: Destination of each flow
        A: Link adjacency matrix (0, 1),   B: Capacity matrix,  mu: Monitoring traffic ratio to total traffic
        R: Routing Matrix  ---> BLP Variable"""
    def __init__(self, in_topo=None):
        self.L = 60  # Maximum length of monitoring routes
        self.x = 1  # x monitoring flows will cross each link
        self.mu = 0.01  # max ratio of monitoring traffic to link capacity
        self.__convert_topo_to_desigred_format(in_topo) # N, number_of_hosts, A, map_switch_to_MAC, possible_sources, map_sourceSwitch_to_host
        self.B = 10000000 * self.A  # links are considered to be 10Mb
        ''' flows adjustment'''
        self.F = self.number_of_links_between_switches*self.x  # Number of Flows
        temp = len(self.possible_sources)
        self.src = [self.possible_sources[i%temp] for i in range(self.F)]  # Sources of each flow
        self.dst = self.src  # Destination of each flow
        self.flow_rate = 1  # monitoring flows are considered to be 1Mbps
        self.__forwarding_table_entries = None
        self.monitoring_lp_problem = None
        self.routing_matrix = [[[0 for f in range(self.F)] for j in range(self.N)] for i in range(self.N)]
        self.big_number = 51*(self.N*self.N)-150*self.N+100
    def __convert_topo_to_desigred_format(self, topo):
        """ """
        if topo is None: topo = {('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 's'): 0, ('7a:1b:c3:65:e5:4a', '00:00:00:00:00:00:00:02', 'h'): 1, ('96:47:ee:7d:c4:3e', '00:00:00:00:00:00:00:01', 'h'): 1, ('ce:74:73:a5:e7:53', '00:00:00:00:00:00:00:03', 'h'): 1}
        '''Find the set of switches and hosts'''
        set_of_switchs, set_of_hosts = set(), set()
        def is_MAC(input):
            if len(input.split(':')) == 8: return True
            else: return False
        for element in topo:
            # each "element" is either (MAC, MAC, 's') OR (IP, MAC, 'h')
            for i in range(0,2):
                if is_MAC(element[i]): set_of_switchs.add(element[i])
                else: set_of_hosts.add(element[i])
        ''' Set parameters --> N: number of nodes (switches),  H: number of hosts, A: adjacency matrix'''
        number_of_links_between_switches = 0
        A = np.zeros((len(set_of_switchs), len(set_of_switchs)))
        # A = [[0 for j in range(len(set_of_switchs))] for i in range(len(set_of_switchs))]
        map_MAC_to_switch = {}
        tN, number_of_hosts = 0, 0  # tN: number of different IPs viewed till now
        possible_sources = set()  # s: set of switches with possibility to be a source (connected to a host)
        map_sourceSwitch_to_host = {}
        for element in topo:
            value = topo[element]
            ''' if both nodes are switch'''
            if element[2] == 's':
                # each "element" is either (MAC, MAC, 's') OR (IP, MAC, 'h')
                for i in range(0, 2):
                    if element[i] not in map_MAC_to_switch:
                        map_MAC_to_switch[element[i]] = tN
                        tN += 1
                if int(value) == 1:
                    # A[map_MAC_to_switch[element[0]], map_MAC_to_switch[element[1]]] = 1
                    A[map_MAC_to_switch[element[0]]][map_MAC_to_switch[element[1]]] = 1
                    number_of_links_between_switches += 1
            # ''' if one node is a switch and another one is a host '''
            elif element[2] == 'h':
                number_of_hosts += 1
                switch = element[1] if is_MAC(element[1]) else element[0]
                host = element[0] if is_MAC(element[1]) else element[1]
                ''' if the connected switch doesn't have a mapping number (from IP to switch number) then assign one'''
                if switch not in map_MAC_to_switch:
                    map_MAC_to_switch[switch] = tN
                    tN += 1
                ''' add the switch number to set of possible sources'''
                possible_sources.add(map_MAC_to_switch[switch])
                ''' map the host to to a source switch'''
                if switch in map_sourceSwitch_to_host:
                    map_sourceSwitch_to_host[switch].append(host)
                else:
                    map_sourceSwitch_to_host[switch] = [host]
        map_switch_to_MAC = {}
        for element in map_MAC_to_switch:
            map_switch_to_MAC[map_MAC_to_switch[element]] = element

        self.number_of_links_between_switches = number_of_links_between_switches
        self.N, self.number_of_hosts, self.A, self.map_switch_to_MAC, self.possible_sources, self.map_sourceSwitch_to_host = \
            tN, number_of_hosts, A, map_switch_to_MAC, list(possible_sources), map_sourceSwitch_to_host
    def __routing_matrix(self):
        """"""
        ''' to simplify the writing'''
        F, N, src, dst, solved_problem = self.F, self.N, self.src, self.dst, self.monitoring_lp_problem
        def index_extractor(input):
            """ remove waste characters """
            input = input.replace('(', '').replace(',', '').replace(')', '')
            return int(str(input).split('_')[1]), int(str(input).split('_')[2]), int(str(input).split('_')[3])
        ''' create route-matrix'''
        routing_matrix = [[[0 for f in range(F)] for j in range(N)] for i in range(N)]
        for variable in solved_problem.variables():
            if int(variable.varValue) is not 0:
                if 'R_(' in variable.name:
                    x, y, z = index_extractor(variable.name)
                    routing_matrix[x][y][z] = int(variable.varValue)
                    self.routing_matrix[x][y][z] = max(routing_matrix[x][y][z], self.routing_matrix[x][y][z])
        return routing_matrix
    def __convert_to_routing_rule_entries(self):
        """"""
        ''' to simplify the writing'''
        F, N, src, dst, solved_problem, map_sourceSwitch_to_host, map_switch_to_MAC = \
            self.F, self.N, self.src, self.dst, self.monitoring_lp_problem, self.map_sourceSwitch_to_host, self.map_switch_to_MAC

        routing_matrix = self.__routing_matrix()
        # def index_extractor(input):
        #     """ remove waste characters """
        #     input = input.replace('(', '').replace(',', '').replace(')', '')
        #     return int(str(input).split('_')[1]), int(str(input).split('_')[2]), int(str(input).split('_')[3])
        def find_next_hob(f, i):
            for j in range(N):
                if routing_matrix[i][j][f] is not 0:
                    return j
            ''' if there is not any next hob then return current hob'''
            return -1
        # ''' create route-matrix'''
        # routing_matrix = [[[0 for f in range(F)] for j in range(N)] for i in range(N)]
        # for variable in solved_problem.variables():
        #     if int(variable.varValue) is not 0:
        #         if 'R_(' in variable.name:
        #             x, y, z = index_extractor(variable.name)
        #             routing_matrix[x][y][z] = int(variable.varValue)

        ''' create routes-rule-entries'''
        routes_rule_entries = [[] for f in range(F)]
        for f in range(F):
            tmp_src = src[f]
            # add source host
            routes_rule_entries[f].append(map_sourceSwitch_to_host[map_switch_to_MAC[tmp_src]][0])
            # add switches in the route
            routes_rule_entries[f].append(map_switch_to_MAC[tmp_src])
            ''' In some cases the source and destination are the same (loop): find next hob '''
            tmp_src = find_next_hob(f, tmp_src)
            # if the next hob is not -1 (meaning there is not any next hob) add that to route-rules
            if tmp_src is not -1:
                routes_rule_entries[f].append(map_switch_to_MAC[tmp_src])
            while tmp_src is not dst[f] and tmp_src is not -1:
                ''' find next hob '''
                tmp_src = find_next_hob(f, tmp_src)
                # if next-hob is -1 means there is not any next hob
                if tmp_src is not -1:
                    routes_rule_entries[f].append(map_switch_to_MAC[tmp_src])
            #add destination host
            routes_rule_entries[f].append(map_sourceSwitch_to_host[map_switch_to_MAC[dst[f]]][0])
        if self.__forwarding_table_entries is None:
            self.__forwarding_table_entries = routes_rule_entries
        else:
            # merge new and existing forwarding table entries
            garbage_output = [self.__forwarding_table_entries.append(entry) for entry in routes_rule_entries]
    def __purge_redundant_flows(self):
        tempF = len(self.__forwarding_table_entries)
        for f in range(tempF-1, -1 , -1):
            ''' removing flows doesn't leave the source switch'''
            if len(self.__forwarding_table_entries[f]) == 3 or (len(self.__forwarding_table_entries[f]) == 4 and self.__forwarding_table_entries[f][1] == self.__forwarding_table_entries[f][2]):
                self.F -= 1
                del self.__forwarding_table_entries[f]
                for i in range(self.N):
                    for j in range(self.N):
                        del self.routing_matrix[i][j][f]
    def solve_optimal(self):
        """ Three steps: 1. Create Problem,  2. Define Variables, and 3. Define Constraints"""
        ''' Step 1. create a (I)LP problem '''
        monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpProblem)

        ''' to simplify the writing '''
        N, F, src, dst, A, mu, L, flow_rate, B, x = self.N, self.F, self.src, self.dst, self.A, self.mu, self.L, self.flow_rate, self.B, self.x

        '''Step 2. define (I)LP variables --> cat={'Binary', 'Continuous', 'Integer'} '''
        R = pulp.LpVariable.dicts('R', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), cat='Binary')
        P = pulp.LpVariable.dicts('P', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), lowBound=0, upBound=L + 1, cat='Integer')
        # Y = pulp.LpVariable('Y',lowBound=0,upBound=1,cat='Binary')
        Y = pulp.LpVariable.dicts('Y', ((f, f_prime) for f in range(F) for f_prime in range(F)), cat='Binary') #meta-variable used to make sure there is not any repeated path (similar path for two probes)

        '''Objective function'''
        monitoring_lp_problem += R[0, 0, 0]
        ''' -1 for MAX the objective function and 1 for minimization of that'''
        monitoring_lp_problem.sense = 1

        '''Step 3.  Define constraints '''
        '''****************************** DO NOT USE < OR > *****************************
        ************************ JUST DO USE >= OR <= OR == *****************************
        ************** NUMPY VALUES ARE DIFFERENT FROM SIMPLE VALUES ********************
        ******************** NUMPY.INT64(0) is not the same as 0 *********************'''
        ''' Don'nt use links that doesn't exist'''
        for i in range(N):
            for j in range(N):
                for f in range(F):
                    monitoring_lp_problem += R[i, j, f] <= int(A[i][j])
        ''' From each link, at least x flows pass'''
        for i in range(N):
            for j in range(N):
                if int(A[i][j]) is not 0:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) >= x
        ''' Flow conservation for R'''
        for f in range(F):
            for i in range(N):
                if src[f] == dst[f] == i:
                    # in this case, loop is allowed
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
                elif i == src[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 1
                elif i == dst[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == -1
                else:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
        ''' Use at most miu percent of bandwidth for monitoring'''
        for i in range(N):
            for j in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) * flow_rate <= mu * float(B[i][j])
        ''' No loop (except for case src[f]==dst[f]): do not leave a node twice'''
        for f in range(F):
            for i in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) <= 1  # circle
        ''' Keep the length of monitoring routes less than a predefined value'''
        for f in range(F):
            monitoring_lp_problem += pulp.lpSum([R[i, j, f] for i in range(N) for j in range(N)]) <= L
        ''' If R[i, j, f] is zero then P[i, j, f] is zero'''
        for f in range(F):
            for i in range(N):
                for j in range(N):
                    monitoring_lp_problem += P[i, j, f] <= (L + 1) * R[i, j, f]
        ''' Add the value of P by one, after each step'''
        for f in range(F):
            for i in range(N):
                if not i == src[f] == dst[f]:
                    monitoring_lp_problem += lpSum([P[i, j, f] for j in range(N)]) == lpSum(
                        [P[j, i, f] + R[i, j, f] for j in range(N)])
        ''' Leave source node and enter destination node (to make sure there is a path for each flow, otherwise, some flows will not come out of source)'''
        for f in range(F):
            monitoring_lp_problem += lpSum([R[src[f], j, f] for j in range(N)]) == 1
            monitoring_lp_problem += lpSum([R[i, dst[f], f] for i in range(N)]) == 1
        ''' No repeated path (n-equation m-unknown where n<m has infinite answers, n>m has no answer, n==m has one answer'''
        for f in range(F):
            for f_prime in range(f+1, F):
                monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) <= Y[f, f_prime]*self.big_number
                monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) >= -Y[f, f_prime]*self.big_number
                # monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) <= -1 + (1-Y[f, f_prime])*100*self.big_number
                monitoring_lp_problem += pulp.lpSum([(100*i+j)*R[i, j, f] for i in range(N) for j in range(N)])-pulp.lpSum([(100*i+j)*R[i, j, f_prime] for i in range(N) for j in range(N)]) >= 1 -Y[f, f_prime]*100*self.big_number
        self.monitoring_lp_problem = monitoring_lp_problem

        # self.monitoring_lp_problem.setSolver(pulp.GLPK_CMD)
        # self.monitoring_lp_problem.solve(pulp.GLPK_CMD(msg=1, options=["--tmlim", "120"]))

        self.monitoring_lp_problem.solve()

        # self.monitoring_lp_problem.parameters.timelimit.set(300.0)

        # P = monitoring_lp_problem
        # Build the solverModel for your preferred
        # solver = pulp.GLPK_CMD()
        # solver.buildSolverModel(P)

        #Modify the solvermodel
        # solver.solverModel.parameters.timelimit.set(60)

        #Solve P
        # solver.callSolver(P)
        # status = solver.findSolutionValues(P)

    def solve_for_some_links(self, links_to_be_crossed, number_of_flows):
        """ Three steps: 1. Create Problem,  2. Define Variables, and 3. Define Constraints"""
        ''' Step 1. create a (I)LP problem '''
        monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpProblem)

        ''' to simplify the writing '''
        N, src, dst, A, mu, flow_rate, L, flow_rate, B, x = self.N, self.src, self.dst, self.A, self.mu, self.flow_rate, self.L, self.flow_rate, self.B, self.x
        F = number_of_flows

        '''Step 2. define (I)LP variables --> cat={'Binary', 'Continuous', 'Integer'} '''
        R = pulp.LpVariable.dicts('R', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), cat='Binary')
        P = pulp.LpVariable.dicts('P', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), lowBound=0, upBound=L + 1, cat='Integer')
        # P shows the step of each movement -- using this variable we prevent flows from making a isolated loop along with the main route from source to destination

        '''Objective function'''
        monitoring_lp_problem += R[0, 0, 0]
        ''' -1 for MAX the objective function and 1 for minimization of that'''
        monitoring_lp_problem.sense = 1

        '''Step 3.  Define constraints '''
        '''****************************** DO NOT USE < OR > *****************************
        ************************ JUST DO USE >= OR <= OR == *****************************'''
        ''' Don'nt use links that doesn't exist'''
        for i in range(N):
            for j in range(N):
                for f in range(F):
                    monitoring_lp_problem += R[i, j, f] <= int(A[i, j])
        ''' From each link, exactly x flows pass'''
        for i in range(N):
            for j in range(N):
                if int(A[i, j]) is not 0:
                    if links_to_be_crossed[i,j]:
                        monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) >= x
        ''' Flow conservation for R'''
        for f in range(F):
            for i in range(N):
                if src[f] == dst[f] == i:
                    # in this case, loop is allowed
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
                elif i == src[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 1
                elif i == dst[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == -1
                else:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
        ''' Use at most miu percent of bandwidth for monitoring'''
        for i in range(N):
            for j in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) * flow_rate <= mu * float(B[i, j])
        ''' No loop (except for case src[f]==dst[f]): do not leave a node twice'''
        for f in range(F):
            for i in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) <= 1  # circle
        ''' Keep the length of monitoring routes less than a predefined value'''
        for f in range(F):
            monitoring_lp_problem += pulp.lpSum([R[i, j, f] for i in range(N) for j in range(N)]) <= L
        ''' If R[i, j, f] is zero then P[i, j, f] is zero'''
        for f in range(F):
            for i in range(N):
                for j in range(N):
                    monitoring_lp_problem += P[i, j, f] <= (L + 1) * R[i, j, f]
        ''' Add the value of P by one, after each step'''
        for f in range(F):
            for i in range(N):
                if not i == src[f] == dst[f]:
                    monitoring_lp_problem += lpSum([P[i, j, f] for j in range(N)]) == lpSum(
                        [P[j, i, f] + R[i, j, f] for j in range(N)])
        self.monitoring_lp_problem = monitoring_lp_problem
        self.monitoring_lp_problem.solve()
    def solve_incremental(self):
        for i in range(self.x):
            centinel = 10
            # links_to_be_crossed = self.A.copy()
            links_to_be_crossed = [[int(self.A[i,j]) for j in range(self.N)] for i in range(self.N)]
            while centinel>0 and sum([sum(lines) for lines in links_to_be_crossed]):
                print('round {}'.format(11-centinel))
                centinel -= 1
                self.__solve_and_maximize_passed_links(links_to_be_crossed, number_of_flows=3)
                if self.optimality_status() is 'Optimal':
                    #remove links that are crossed already from links_to_be_crossed
                    routing_matrix = self.__routing_matrix()
                    links_to_be_crossed = [[int(links_to_be_crossed[i][j] and (not sum(routing_matrix[i][j]))) for j in range(self.N)] for i in range(self.N)]
                    #update routing rules
                    self.__convert_to_routing_rule_entries()
                    # self.__purge_redundant_flows()
                else: raise Exception('Problem is not solvable')
        self.__purge_redundant_flows()
    def __solve_and_maximize_passed_links(self, links_to_be_crossed, number_of_flows):
        """ Three steps: 1. Create Problem,  2. Define Variables, and 3. Define Constraints"""
        ''' Step 1. create a (I)LP problem '''
        monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpProblem)

        ''' to simplify the writing '''
        N, src, dst, A, mu, flow_rate, L, flow_rate, B, x = self.N, self.src, self.dst, self.A, self.mu, self.flow_rate, self.L, self.flow_rate, self.B, self.x
        F = number_of_flows

        '''Step 2. define (I)LP variables --> cat={'Binary', 'Continuous', 'Integer'} '''
        R = pulp.LpVariable.dicts('R', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), cat='Binary')
        P = pulp.LpVariable.dicts('P', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), lowBound=0, upBound=L + 1, cat='Integer')
        # P shows the step of each movement -- using this variable we prevent flows from making a isolated loop along with the main route from source to destination

        '''Objective function'''
        monitoring_lp_problem += pulp.lpSum([R[i, j, f]*links_to_be_crossed[i][j] for i in range(N) for j in range(N) for f in range(F)])
        ''' -1 for MAX the objective function and 1 for minimization of that'''
        monitoring_lp_problem.sense = -1

        '''Step 3.  Define constraints '''
        '''****************************** DO NOT USE < OR > *****************************
        ************************ JUST DO USE >= OR <= OR == *****************************'''
        ''' Don't use links that doesn't exist'''
        for i in range(N):
            for j in range(N):
                for f in range(F):
                    monitoring_lp_problem += R[i, j, f] <= int(A[i, j])
        ''' From each link, exactly x flows pass'''
        for i in range(N):
            for j in range(N):
                if int(A[i, j]) is not 0:
                    if links_to_be_crossed[i,j]:
                        monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) >= x
        ''' Flow conservation for R'''
        for f in range(F):
            for i in range(N):
                if src[f] == dst[f] == i:
                    # in this case, loop is allowed
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
                elif i == src[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 1
                elif i == dst[f]:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == -1
                else:
                    monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) - pulp.lpSum(
                        [R[j, i, f] for j in range(N)]) == 0
        ''' Use at most miu percent of bandwidth for monitoring'''
        for i in range(N):
            for j in range(N):
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) * flow_rate <= mu * float(B[i, j])
        ''' No loop (except for case src[f]==dst[f]): do not leave a node twice'''
        # for f in range(F):
        #     for i in range(N):
        #         monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) <= 1  # circle
        ''' Keep the length of monitoring routes less than a predefined value (it is required for next constraint which is CNST2'''
        for f in range(F):
            monitoring_lp_problem += pulp.lpSum([R[i, j, f] for i in range(N) for j in range(N)]) <= L
        ''' CNST2: If R[i, j, f] is zero then P[i, j, f] is zero'''
        for f in range(F):
            for i in range(N):
                for j in range(N):
                    monitoring_lp_problem += P[i, j, f] <= (L + 1) * R[i, j, f]
        ''' Add the value of P by one, after each step'''
        for f in range(F):
            for i in range(N):
                if not i == src[f] == dst[f]:
                    monitoring_lp_problem += lpSum([P[i, j, f] for j in range(N)]) == lpSum(
                        [P[j, i, f] + R[i, j, f] for j in range(N)])
        self.monitoring_lp_problem = monitoring_lp_problem
        self.monitoring_lp_problem.solve()
    def forwarding_table_entries(self):
        if self.monitoring_lp_problem is None:
            raise Exception('Problem is not solved yet')
        if self.optimality_status() is 'Optimal' and self.__forwarding_table_entries is None:
            self.__convert_to_routing_rule_entries()
            self.__purge_redundant_flows()
        return self.__forwarding_table_entries
    def optimality_status(self):
        if monitoring is None: raise Exception('Problem is not defined yet')
        return LpStatus[self.monitoring_lp_problem.status]
    def print_results(self):
        if self.monitoring_lp_problem is not None:
            print('Problem Solving Status: {}'.format(pulp.LpStatus[self.monitoring_lp_problem.status]))
            print("Objective function value: {}".format(pulp.value(self.monitoring_lp_problem.objective)))
            print('Variables: ')
            for variable in self.monitoring_lp_problem.variables():
                if int(variable.varValue) is not 0:
                    print("   {} = {}".format(variable.name, variable.varValue))
        else:
            raise Exception('The problem is not solved yet.')
if __name__ == '__main__':
    # main()
    now = datetime.datetime.now()
    # topo = {('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', 's'): 1,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', 's'): 1,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:03', 's'): 1,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:09', 's'): 1,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:02', 's'): 1,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:04', 's'): 1,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:03', 's'): 1,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:05', 's'): 1,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0d', 's'): 1,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:04', 's'): 1,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:06', 's'): 1,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0d', 's'): 1,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:05', 's'): 1,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:07', 's'): 1,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:15', 's'): 1,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:16', 's'): 1,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:06', 's'): 1,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:08', 's'): 1,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:10', 's'): 1,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:07', 's'): 1,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:10', 's'): 1,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:02', 's'): 1,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:11', 's'): 1,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:1b', 's'): 1,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:01', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:02', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:03', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:11', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:12', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:13', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:14', 's'): 1,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:03', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:04', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:12', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:13', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:14', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:15', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:16', 's'): 1,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:04', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:05', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0d', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:1a', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:1b', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:1c', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:1d', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:1e', 's'): 1,('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:04', 's'): 1,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:05', 's'): 1,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:17', 's'): 1,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:1d', 's'): 1,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:05', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:07', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0d', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:17', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:18', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:1a', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:1e', 's'): 1,('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:06', 's'): 1,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:07', 's'): 1,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:08', 's'): 1,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:16', 's'): 1,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:17', 's'): 1,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:18', 's'): 1,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:1f', 's'): 1,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:07', 's'): 1,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:08', 's'): 1,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:17', 's'): 1,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:18', 's'): 1,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:09', 's'): 1,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:19', 's'): 1,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:1a', 's'): 1,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:1b', 's'): 1,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:13', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:0a', 's'): 1,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:1d', 's'): 1,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:14', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:06', 's'): 1,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:1b', 's'): 1,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:15', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:06', 's'): 1,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:0b', 's'): 1,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:1e', 's'): 1,('00:00:00:00:00:00:00:16', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:0d', 's'): 1,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:10', 's'): 1,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:1e', 's'): 1,('00:00:00:00:00:00:00:17', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:10', 's'): 1,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:18', '00:00:00:00:00:00:00:1f', 's'): 1,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:11', 's'): 1,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:1a', 's'): 1,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:19', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:12', 's'): 1,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:19', 's'): 1,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:1b', 's'): 1,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:1a', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:09', 's'): 1,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:13', 's'): 1,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:15', 's'): 1,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:1a', 's'): 1,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:1c', 's'): 1,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:1b', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:1b', 's'): 1,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:1d', 's'): 1,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:1c', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:0d', 's'): 1,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:14', 's'): 1,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:1c', 's'): 1,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:1e', 's'): 1,('00:00:00:00:00:00:00:1d', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:0c', 's'): 1,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:0e', 's'): 1,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:0f', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:16', 's'): 1,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:17', 's'): 1,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:18', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:1d', 's'): 1,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:1e', 's'): 0,('00:00:00:00:00:00:00:1e', '00:00:00:00:00:00:00:1f', 's'): 1,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:01', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:02', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:03', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:04', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:05', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:06', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:07', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:08', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:09', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:0a', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:0b', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:0c', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:0d', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:0e', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:0f', 's'): 1,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:10', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:11', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:12', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:13', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:14', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:15', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:16', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:17', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:18', 's'): 1,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:19', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:1a', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:1b', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:1c', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:1d', 's'): 0,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:1e', 's'): 1,('00:00:00:00:00:00:00:1f', '00:00:00:00:00:00:00:1f', 's'): 0,('00:00:00:00:00:01', '00:00:00:00:00:00:00:19', 'h'): 1,('00:00:00:00:00:02', '00:00:00:00:00:00:00:1a', 'h'): 1,('00:00:00:00:00:03', '00:00:00:00:00:00:00:1b', 'h'): 1,('00:00:00:00:00:04', '00:00:00:00:00:00:00:1c', 'h'): 1,('00:00:00:00:00:05', '00:00:00:00:00:00:00:1d', 'h'): 1,('00:00:00:00:00:06', '00:00:00:00:00:00:00:1e', 'h'): 1,('00:00:00:00:00:07', '00:00:00:00:00:00:00:1f', 'h'): 1,('00:00:00:00:00:08', '00:00:00:00:00:00:00:18', 'h'): 1,('00:00:00:00:00:09', '00:00:00:00:00:00:00:10', 'h'): 1,('00:00:00:00:00:10', '00:00:00:00:00:00:00:08', 'h'): 1,('00:00:00:00:00:11', '00:00:00:00:00:00:00:07', 'h'): 1,('00:00:00:00:00:12', '00:00:00:00:00:00:00:06', 'h'): 1,('00:00:00:00:00:13', '00:00:00:00:00:00:00:05', 'h'): 1,('00:00:00:00:00:14', '00:00:00:00:00:00:00:04', 'h'): 1,('00:00:00:00:00:15', '00:00:00:00:00:00:00:03', 'h'): 1,('00:00:00:00:00:16', '00:00:00:00:00:00:00:02', 'h'): 1,('00:00:00:00:00:17', '00:00:00:00:00:00:00:01', 'h'): 1}
    topo = {('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:05', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:05', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:05', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 'h'): 1, ('00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 'h'): 1, ('00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 'h'): 1, ('00:00:00:00:00:04', '00:00:00:00:00:00:00:04', 'h'): 1, ('00:00:00:00:00:05', '00:00:00:00:00:00:00:05', 'h'): 1, ('00:00:00:00:00:06', '00:00:00:00:00:00:00:0a', 'h'): 1, ('00:00:00:00:00:07', '00:00:00:00:00:00:00:09', 'h'): 1, ('00:00:00:00:00:08', '00:00:00:00:00:00:00:08', 'h'): 1}
    # topo = None
    x = monitoring(in_topo=topo)
    x.solve_optimal()
    # x.solve_incremental()

    optimality_status = x.optimality_status()
    forwarding_table_entries = x.forwarding_table_entries()
    routing_matrix = x.routing_matrix

    print("Solution status: {}".format(optimality_status))
    print('Execution time: ' + str(datetime.datetime.now() - now))
    print(forwarding_table_entries)
    print(routing_matrix)
    print(x.map_switch_to_MAC)
