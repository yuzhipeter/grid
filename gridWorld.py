###~~~~~~~~~~~~~
# Notes: This example is modified from
# https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4
###~~~~~~~~~~~~~

import sys
sys.path.append(r"./burlap.jar")

from copy import deepcopy
from collections import defaultdict
from collections import deque
from time import time
import csv
import pickle
import java

from burlap.behavior.policy import Policy;
from burlap.assignment4 import BasicGridWorld;
from burlap.behavior.singleagent import EpisodeAnalysis;
from burlap.behavior.singleagent.auxiliary import StateReachability;
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.behavior.valuefunction import ValueFunction;
from burlap.domain.singleagent.gridworld import GridWorldDomain;
from burlap.oomdp.core import Domain;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent import SADomain;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.statehashing import HashableStateFactory;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent.explorer import VisualExplorer;
from burlap.oomdp.visualizer import Visualizer;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.assignment4.EasyGridWorldLauncher import visualizeInitialGridWorld
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis,getAllStates
from burlap.behavior.learningrate import ExponentialDecayLR, SoftTimeInverseDecayLR

def vIteration(world, userMap, maxX, maxY, discount=0.99, MAX_ITERATIONS=100):
    gen = BasicGridWorld(userMap, maxX, maxY)
    domain = gen.generateDomain()
    initialState = gen.getExampleState(domain);

    rf = BasicRewardFunction(maxX, maxY, userMap)
    tf = BasicTerminalFunction(maxX, maxY)
    env = SimulatedEnvironment(domain, rf, tf, initialState)
    visualizeInitialGridWorld(domain, gen, env)

    hashingFactory = SimpleHashableStateFactory()
    timing = defaultdict(list)
    rewards = defaultdict(list)
    steps = defaultdict(list)
    convergence = defaultdict(list)

    allStates = getAllStates(domain, rf, tf, initialState)

    print("*** {} Value Iteration Analysis".format(world))

    MAX_ITERATIONS = MAX_ITERATIONS
    iterations = range(1, MAX_ITERATIONS + 1)
    vi = ValueIteration(domain, rf, tf, discount, hashingFactory, -1, 1);
    vi.setDebugCode(0)
    vi.performReachabilityFrom(initialState)
    vi.toggleUseCachedTransitionDynamics(False)
    timing['Value'].append(0)
    for nIter in iterations:
        startTime = clock()
        vi.runVI()
        p = vi.planFromState(initialState);
        endTime = clock()
        timing['Value'].append((endTime-startTime)*1000)

        convergence['Value'].append(vi.latestDelta)
        # evaluate the policy with evalTrials roll outs
        runEvals(initialState, p, rewards['Value'], steps['Value'], rf, tf, evalTrials=1)
        if nIter == 1 or nIter == 50:
            simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration {}".format(nIter))

    simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration {}".format(nIter))
    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
            world + ' Value Iteration Policy Map.pkl')
    dumpCSV(nIter, timing['Value'][1:], rewards['Value'], steps['Value'], convergence['Value'], world, 'Value')


def pIteration(world, userMap, maxX, maxY, discount=0.99, MAX_ITERATIONS=100):
    gen = BasicGridWorld(userMap, maxX, maxY)
    domain = gen.generateDomain()
    initialState = gen.getExampleState(domain);

    rf = BasicRewardFunction(maxX, maxY, userMap)
    tf = BasicTerminalFunction(maxX, maxY)
    env = SimulatedEnvironment(domain, rf, tf, initialState)
    visualizeInitialGridWorld(domain, gen, env)

    hashingFactory = SimpleHashableStateFactory()
    timing = defaultdict(list)
    rewards = defaultdict(list)
    steps = defaultdict(list)
    convergence = defaultdict(list)
    policy_converged = defaultdict(list)    
    last_policy = defaultdict(list)

    allStates = getAllStates(domain, rf, tf, initialState)

    print("*** {} Policy Iteration Analysis".format(world))

    MAX_ITERATIONS = MAX_ITERATIONS
    iterations = range(1, MAX_ITERATIONS + 1)
    pi = PolicyIteration(domain,rf,tf,discount,hashingFactory,-1,1,1); 
    pi.setDebugCode(0)
    for nIter in iterations:
        startTime = clock()
        #pi = PolicyIteration(domain,rf,tf,discount,hashingFactory,-1,1, nIter); 
        #pi.setDebugCode(0)
        # run planning from our initial state
        p = pi.planFromState(initialState);
        endTime = clock()
        timing['Policy'].append((endTime-startTime)*1000)

        convergence['Policy'].append(pi.lastPIDelta)         
        # evaluate the policy with one roll out visualize the trajectory
        runEvals(initialState, p, rewards['Policy'], steps['Policy'], rf, tf, evalTrials=1)
        if nIter == 1 or nIter == 50:
            simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration{}".format(nIter))
 
        policy = pi.getComputedPolicy()
        allStates = pi.getAllStates()
        current_policy = [[(action.ga, action.pSelection) 
            for action in policy.getActionDistributionForState(state)] 
            for state in allStates]
        policy_converged['Policy'].append(current_policy == last_policy)
        last_policy = current_policy
 
    simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration{}".format(nIter))
    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
            world + ' Policy Iteration Policy Map.pkl')
    dumpCSVp(iterations, timing['Policy'], rewards['Policy'], steps['Policy'],convergence['Policy'], 
            world, 'Policy', policy_converged['Policy'])


def qLearning(world, userMap, maxX, maxY, discount=0.9, MAX_ITERATIONS=1000):
    gen = BasicGridWorld(userMap, maxX, maxY)
    domain = gen.generateDomain()
    initialState = gen.getExampleState(domain);

    rf = BasicRewardFunction(maxX, maxY, userMap)
    tf = BasicTerminalFunction(maxX, maxY)
    env = SimulatedEnvironment(domain, rf, tf, initialState)
    visualizeInitialGridWorld(domain, gen, env)

    hashingFactory = SimpleHashableStateFactory()
    timing = defaultdict(list)
    rewards = defaultdict(list)
    steps = defaultdict(list)
    convergence = defaultdict(list)

    allStates = getAllStates(domain, rf, tf, initialState)

    MAX_ITERATIONS = MAX_ITERATIONS
    NUM_INTERVALS = MAX_ITERATIONS;
    iterations = range(1, MAX_ITERATIONS + 1)
    qInit = 0
    for lr in [0.01, 0.1, 0.5]:
        for epsilon in [0.3, 0.5, 0.7]:
            last10Chg = deque([10] * 10, maxlen=10)
            Qname = 'Q-Learning L{:0.2f} E{:0.1f}'.format(lr, epsilon)
            #agent = QLearning(domain, discount, hashingFactory, qInit, lr, epsilon, 300)
            agent = QLearning(domain, discount, hashingFactory, qInit, lr, epsilon)
            agent.setDebugCode(0)

            print("*** {}: {}".format(world, Qname))

            for nIter in iterations:
                if nIter % 200 == 0: 
                    print('Iteration: {}'.format(nIter))

                startTime = clock()
                #ea = agent.runLearningEpisode(env, 300)
                ea = agent.runLearningEpisode(env)
                env.resetEnvironment()
                agent.initializeForPlanning(rf, tf, 1)
                p = agent.planFromState(initialState)  # run planning from our initial state
                endTime = clock()
                timing[Qname].append((endTime-startTime)*1000)

                last10Chg.append(agent.maxQChangeInLastEpisode)
                convergence[Qname].append(sum(last10Chg)/10.)
                # evaluate the policy with one roll out visualize the trajectory
                runEvals(initialState, p, rewards[Qname], steps[Qname], rf, tf, evalTrials=1)
                if nIter % 1000 == 0:
                    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                                  '{} {} Iter {} Policy Map.pkl'.format(world, Qname, nIter))
                    simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, Qname)
                
            dumpCSV(nIter, timing[Qname], rewards[Qname], steps[Qname], convergence[Qname], world, Qname) 

def dumpCSV(nIter, times,rewards,steps,convergence,world,method):
    fname = '{} {}.csv'.format(world,method)
    iters = range(1, nIter + 1)
    assert len(iters)== len(times)
    assert len(iters)== len(rewards)
    assert len(iters)== len(steps)
    assert len(iters)== len(convergence)
    with open(fname,'wb') as f:
        f.write('iter,time,reward,steps,convergence,policy\n')
        writer = csv.writer(f,delimiter=',')
        writer.writerows(zip(iters,times,rewards,steps,convergence))

def dumpCSVp(iters, times,rewards,steps,convergence,world,method,policy):
    fname = '{} {}.csv'.format(world, method)
    assert len(iters)== len(times)
    assert len(iters)== len(rewards)
    assert len(iters)== len(steps)
    assert len(iters)== len(convergence)
    assert len(iters)== len(policy)
    with open(fname,'wb') as f:
        f.write('iter,time,reward,steps,convergence,policy\n')
        writer = csv.writer(f,delimiter=',')
        writer.writerows(zip(iters,times,rewards,steps,convergence,policy))
    
def runEvals(initialState,plan,rewardL,stepL, rf, tf, evalTrials):
    r = []
    s = []
    for trial in range(evalTrials):
        ea = plan.evaluateBehavior(initialState, rf, tf,50000);
        r.append(calcRewardInEpisode(ea))
        s.append(ea.numTimeSteps())
    rewardL.append(sum(r)/float(len(r)))
    stepL.append(sum(s)/float(len(s))) 

def comparePolicies(policy1, policy2):
    assert len(policy1) == len(policy1)
    diffs = 0
    for k in policy1.keys():
        if policy1[k] != policy2[k]:
            diffs += 1
    return diffs

def mapPicture(javaStrArr):
    out = []
    for row in javaStrArr:
        out.append([])
        for element in row:
            out[-1].append(str(element))
    return out

def dumpPolicyMap(javaStrArr, fname):
    pic = mapPicture(javaStrArr)
    with open(fname, 'wb') as f:
        pickle.dump(pic, f)

def gridworld(world = 'Easy'):
    if world == 'Easy':
        #userMap = [[-4, -4, -4, -4, 100],
        #           [-4, 1, -4, 1, -100],
        #           [-4, 1, 1, 1, -4],
        #           [-4, 1, -4, 1, -4],
        #           [-4, -4, -4, -4, -4]]
        #userMap = [[1,0,0,0],
        #       [0,1,0,0],
        #       [0,1,1,0],
        #       [0,0,0,0]]
        userMap = [[0,1,0,0,0],
                   [0,1,0,1,0],
                   [0,1,0,0,0],
                   [0,1,1,1,0],
                   [0,0,0,0,0]]
    else:
        userMap = [[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
               [1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
               [1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
               [1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
               [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
               [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
               [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
               [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
               [0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0],
               [0,0,0,1,1,0,0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    n = len(userMap)
    tmp = deepcopy(userMap)
    userMap = MapPrinter().mapToMatrix(tmp)
    maxX = n - 1
    maxY = n - 1
    
    # Print the map that is being analyzed
    print("\n\n*** {} Grid World Analysis ***\n".format(world))
    MapPrinter().printMap(MapPrinter.matrixToMap(userMap));
    
    return userMap, maxX, maxY


###~~~~~~~~~~~~~~~~~~~~~~~~~
### run
###~~~~~~~~~~~~~~~~~~~~~~~~~
worlds = ['Easy', 'Hard']
for world in worlds:
    userMap, maxX, maxY = gridworld(world = world)
    vIteration(world, userMap, maxX, maxY, discount=0.99, MAX_ITERATIONS=100)
    pIteration(world, userMap, maxX, maxY, discount=0.99, MAX_ITERATIONS=100)
    qLearning(world, userMap, maxX, maxY, discount=0.99, MAX_ITERATIONS=1000)
    
