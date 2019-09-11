from pulp import *
import numpy as np


def read_topology():
    N = 3  # Number of nodes
    # A = np.random.randint(2, size=(N, N))
    A = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])
    # B = 10 * np.random.random((N, N))
    B = 10000 * np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]])
    return N, A, B


def parameter_adjustment():
    mu = 0.01  # Monitoring traffic ratio to total traffic
    L = 100  # Maximum length of monitoring routes
    x = 1  # x monitoring flows will cross each link
    return mu, L, x


def flow_adjustment():
    F = 2  # Number of Flows
    src = [0, 0]  # Sources of each flow
    dst = [0, 0]  # Destination of each flow
    flow_rate = 1
    return F, src, dst, flow_rate


def network_monitoring_ILP(print_problem):
    # --------------------------------------------------------------------------------------------
    # ******************************VERY IMPORTANT INFORMATION:******************************
    print('****************************** DO NOT USE < OR > *****************************')
    print('************************ JUST DO USE >= OR <= OR == **************************')
    print('************** NUMPY VALUES ARE DIFFERENT FROM SIMPLE VALUES *****************')
    print('******************** NUMPY.INT64(0) is not the same as 0 *********************')
    # --------------------------------------------------------------------------------------------
    # create a (I)LP problem with the method LpProblem in PuLP
    # monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpProblem)
    monitoring_lp_problem = pulp.LpProblem("Route Calculation for Monitoring", pulp.LpMinimize)

    # --------------------------------------------------------------------------------------------
    # create ordinary variables
    mu, L, x = parameter_adjustment()
    F, src, dst, flow_rate = flow_adjustment()
    N, A, B = read_topology()

    # --------------------------------------------------------------------------------------------
    # create (I)LP variables --> cat={'Binary', 'Continuous', 'Integer',
    R = pulp.LpVariable.dicts('R', ((i, j, f) for i in range(N) for j in range(N) for f in range(F)), cat='Binary')
    # mu = pulp.LpVariable('mu', lowBound=0, upBound=1, cat='Continuous')

    # --------------------------------------------------------------------------------------------
    # Objective function
    monitoring_lp_problem += R[0, 0, 0]
    ''' -1 for MAX the objective function and 1 for minimization of that'''
    monitoring_lp_problem.sense = 1

    # --------------------------------------------------------------------------------------------
    # Set the constraints
    ''' Don'nt use links that doesn't exist'''
    for i in range(N):
        for j in range(N):
            for f in range(F):
                monitoring_lp_problem += R[i, j, f] <= int(A[i, j])
    ''' From each link, exactly x flows pass'''
    for i in range(N):
        for j in range(N):
            if int(A[i, j]) is not 0:
                monitoring_lp_problem += pulp.lpSum([R[i, j, f] for f in range(F)]) >= x
    ''' Flow conservation'''
    for f in range(F):
        for i in range(N):
            if src[f] == dst [f] == i:
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
    ''' No loop'''
    for f in range(F):
        for i in range(N):
            monitoring_lp_problem += pulp.lpSum([R[i, j, f] for j in range(N)]) <= 1   # circle
    ''' Keep the length of monitoring routes less than a predefined value'''
    for f in range(F):
        monitoring_lp_problem += pulp.lpSum([R[i, j, f] for i in range(N) for j in range(N)]) <= L

    # --------------------------------------------------------------------------------------------
    # Print down the problem
    if print_problem:
        print(monitoring_lp_problem)

    # --------------------------------------------------------------------------------------------
    # monitoring_lp_problem.solve(pulp.GLPK_CMD())
    monitoring_lp_problem.solve()

    # --------------------------------------------------------------------------------------------
    # printing the results
    print('Problem Solving Status: {}'.format(pulp.LpStatus[monitoring_lp_problem.status]))
    print("Objective function value: {}".format(pulp.value(monitoring_lp_problem.objective)))
    print('Variables: ')
    for variable in monitoring_lp_problem.variables():
        print "   {} = {}".format(variable.name, variable.varValue)


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    network_monitoring_ILP(print_problem=False)
    print datetime.datetime.now()-now
