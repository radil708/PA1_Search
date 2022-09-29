# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def tinyMazeSearch(problem):

    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    @param SearchProblem
    @return a list of Directions objects
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


#custom helper function, do NOT delete
def getActionsFromNodes(last_node : dict) -> list:

    store_actions = []
    store_actions.append(last_node["ACTION"])
    parent_node = last_node["PARENT"]

    while parent_node != None:

        direction_action = parent_node["ACTION"]
        if direction_action is None:
            break
        #TODO delete print statement
        #print(direction_action)
        store_actions.append(direction_action)
        parent_node = parent_node["PARENT"]

    store_actions.reverse()
    return store_actions





def depthFirstSearch(problem: SearchProblem):
    """
    #NEEDS TO RETURN A LIST OF DIRECTIONS TO GO

    Search the deepest nodes in the search tree first.
    # TODO how to access the tree?

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState()) -> Returns a tuple (int,int) -> (x,y) location of agent?
    print("Is the start a goal?", problem.isGoalState(problem.getStartState())) -> Returns Boolean true or false
    print("Start's successors:", problem.getSuccessors(problem.getStartState())) -> Successors are list of child nodes
    [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    First element is a tuple of int so a state?
    Second element is direction which is action?
    Third is maybe some cost??
    """
    "*** YOUR CODE HERE ***"

    #Copied and modified from figure 3.11 AIMA pg. 82


    #uncomment to see tests one at a time
    #input("Hit enter to proceed to next test")

    # set to True to see debug statements
    display = False

    if display:
        print("-----INITIATING DEPTH FIRST SEARCH------")

    ###Initialize the initial-state node: initial node = (initial_state, [])
    # node <- a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    start_node = {}
    start_node["STATE"] = problem.getStartState()
    start_node["COST"] = 0
    start_node["PARENT"] = None
    start_node["ACTION"] = None

    # ##Initialize the frontier
    frontier = util.Stack()

    # Frontier = A LIFO stack with node as the only element
    frontier.push(start_node)


    # Initialize the explored set
    explored = set()

    # loop do
    while True:
        ##if frontier is empty then return Fail
        if frontier.isEmpty():
            raise RuntimeError("No more nodes to explore")

    # curr_node = last node in the frontier, remove curr_node from the frontier
        current_node = frontier.pop()

        if display:
            print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Current State: {}".format(current_node["STATE"]))

        #Add curr_node.state to explored
        # set cannot contain dicts as elements
        explored.add(current_node["STATE"])

        if problem.isGoalState(current_node["STATE"]):
            if display:
                print("\n!!!! GOAL STATE REACHED, DFS COMPLETE !!!!\n")
            return getActionsFromNodes(current_node)
    #
    # 	for each action in problem.ACTIONS(node.STATE) do
        # so get all potential actions i.e. the next nodes
        all_successors = problem.getSuccessors(current_node["STATE"])

        if display:
            print("Current Frontier: {}".format(frontier.list.__str__()))
            print("----------------------------------------------------------------------")

        counter = 0
        for successor_node in all_successors:
            counter += 1
            # child = CHILD-NODE(problem, node, action), also keep track of parent to return a set of directions
            child = {"STATE": successor_node[0], "ACTION": successor_node[1], "COST": successor_node[2],
                     "PARENT": current_node}
            if display:
                print(f"Child# {counter}.) {current_node['STATE']} -> {child['STATE']} at a cost of {child['COST']}")

            #if child.STATE is not in explored or frontier then
            # testing they want us to keep track of node only in explored not frontier
            if child not in frontier.list and child["STATE"] not in explored:
                if display:
                    print("Child State NOT yet explored and NOT in frontier, pushing to frontier")

            #if problem.GOAL-TEST(child.STATE) then return Solution(child)
                frontier.push(child)
                # keep track of path here
                #print(child,current_node)
            else:
                if display:
                    print(f"Child State {child['STATE']} ALREADY explored, going to next child")
            if display:
                print("--------------------------------------------------------------------")
        if display:
            print("TRANSITIONING TO NEXT STATE")


#custom function do NOT delete
def check_queue(queue: list, key: str, expected_val):
    flag = False
    for each in queue:
        if each[key] == expected_val:
            flag = True
            break
    return flag

#custom function do NOT delete
def display_curr_frontier_list(list_in : list, key: str):
    new = []
    for each_dict in list_in:
        new.append(each_dict[key])
    return new


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #uncomment to see tests one at a time, helpful in conjunction with debug statements
    #input("Hit Enter to proceed to next test")

    #set to True to see debug statements
    display = False

    if display:
        print("-----INITIATING BREADTH FIRST SEARCH-----")


    start_node = {"STATE": problem.getStartState(), "PARENT": None,
                  "ACTION": None, "COST": 0}
    frontier = util.Queue()
    explored = []

    frontier.push(start_node)

    while not frontier.isEmpty():
        current_node = frontier.pop()
        explored.append(current_node["STATE"])

        if display:
            print(f"Current State: {current_node['STATE']}")

        if problem.isGoalState(current_node["STATE"]):
            if display:
                print("\n!!!! GOAL STATE REACHED, BFS COMPLETE !!!!")
            break

        if display:
            print(f"Current Frontier: {frontier.list.__str__()}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++")

        counter = 0

        for successor_node in problem.getSuccessors(current_node["STATE"]):
            counter += 1

            child_node = {"STATE": successor_node[0], "ACTION": successor_node[1], "COST": successor_node[2],
                          "PARENT": current_node}
            if display:
                print(f"Child #{counter}.) {child_node['STATE']} via {child_node['ACTION']} at a cost of {child_node['COST']}")

            if successor_node[0] not in explored and not check_queue(frontier.list, "STATE",successor_node[0]):
                if display:
                    print(f"Child state: {child_node['STATE']} NOT in EXPLORED or FRONTIER, push to FRONTIER")
                frontier.push(child_node)
            else:
                if display:
                    print(f"Child State: {child_node['STATE']} ALREADY Explored, going to next node")

            if display:
                print("-----------------------------------------------------------------------------")

        if display:
            print("TRANSITIONING TO NEXT NODE\n")

    return getActionsFromNodes(current_node)


def check_heapq(prioQHeap : list, target_node, priority):
    in_heap = False
    better_prio = False
    print("in checkheap")
    print(prioQHeap)
    for each in prioQHeap:
        if each[2] == target_node:
            in_heap = True
            if priority <= each[0]:
                better_prio = True
    return in_heap == True and better_prio == True

def isAlreadyInHeap(heapIn: list, target):
    in_heap = False
    for each in heapIn:
        # third element in the sublist of heapin is the state
        if each[2] == target:
            in_heap = True
            break
    return in_heap

def hasBetterPrio(heapIn: list, target, priority):
    better_prio = False
    for each in heapIn:
        if each[2] == target and priority <= each[0]:
            better_prio = True
            break
    return better_prio


def getUCSPath(dict_paths: dict, last_node_state, first_node_state):
    path = []

    while last_node_state != first_node_state:
        path.append(dict_paths[last_node_state][1])
        last_node_state = dict_paths[last_node_state][0]

    path.reverse()
    return path


#TODO remove optional arg
def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    #uncomment line below to see tests run one at a time, helpful in conjuction with debug statements
    #input("hit enter to continue")

    #cost of the node will determing priority

    # set display to True to see debug statements, helpful for seeing step by step
    display = False

    #but we need to track parents too....
    child_to_parent_dict = {}
    # key is child node and value is (parent, action)
    child_to_parent_dict[problem.getStartState()] = (None, None)
    first_state = problem.getStartState()
    #initialize start node with state, cost

    shortest_cost_to_state = {problem.getStartState(): 0}


    #initialize prio queue
    frontier = util.PriorityQueue()

    frontier.push(first_state,0)


    #initizalize explored
    explored = []

    if display:
        print("\nNEW TEST\n")

    while not frontier.isEmpty():
        #current node is parent node
        #this only returns the item or "state" not anything else
        current_state = frontier.pop()

        #reached goal!
        if problem.isGoalState(current_state):
            if display:
                print("Goal state has been reached")
            break

        #if we already explored the state, skip it
        if current_state in explored:
            if display:
                print(f"Current State: {current_state} has already been explored")
            continue

        # add to explored
        explored.append(current_state)

        parent_state = current_state

        if display:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            f_list = frontier.heap
            print(f"Current Frontier: {f_list}")
            print(f"Current State is {current_state}")

        counter = 0

        # look through all successor NODES
        for each_child_node in problem.getSuccessors(parent_state):
            child_state = each_child_node[0]

            if child_state in explored:
                if display:
                    print(f"Child State: {child_state} already in explored, going to next node")
                continue

            direction = each_child_node[1]
            cost_to_child_node_from_parent = each_child_node[2]
            cumulative_cost_to_child_node = shortest_cost_to_state[parent_state] + cost_to_child_node_from_parent

            if display:
                counter += 1
                print("----------------------------------------------------------------------------------------------")
                print(f"From {parent_state} to {child_state} go via {direction} at a cost of {cumulative_cost_to_child_node}")

            frontier_list = frontier.heap
            # check if the child is already in frontier
            if isAlreadyInHeap(frontier_list,child_state):
                if display:
                    print(f"{child_state} is already in frontier with a cost of {shortest_cost_to_state[child_state]} "
                          f"and current cost is {cumulative_cost_to_child_node}")

                if hasBetterPrio(frontier_list,child_state,cumulative_cost_to_child_node):

                    if display:
                        print("UPDATING heap with current cost, and child to parent, and shortest cost to child ")

                    frontier.update(child_state,cumulative_cost_to_child_node)
                    # store cost to child node
                    shortest_cost_to_state[child_state] = cumulative_cost_to_child_node
                    # store path from parent to child, key:child value:parent
                    child_to_parent_dict[child_state] = (parent_state, direction)

                    if display:
                        print("----------------------------------------------------------------------------------------------")
                else:
                    continue

            else:
                if display:
                    print(f"child state: {child_state} not in heap, adding to {child_state} to heap")
                #if not in frontier than push with cumulative cost
                frontier.push(child_state,cumulative_cost_to_child_node)
                #store cost to child node
                shortest_cost_to_state[child_state] = cumulative_cost_to_child_node
                #store path from parent to child, key:child value:parent
                child_to_parent_dict[child_state] = (parent_state, direction)

        if display:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")


    if display:
        print("UCS Complete")
    #print(child_to_parent_dict)
    return getUCSPath(child_to_parent_dict,current_state, first_state)





def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    '''
    IMPORTANT:
     - A* takes a heuristic function as an argument.
     - combines UCS and Greedy so keep track of cumulative path cost
     - Heuristics take two arguments: a state in the search problem 
        (the main argument), and the problem itself (for reference information)
     - The nullHeuristic heuristic function in search.py is a trivial example. 
    '''

    '''
    priority of state is f(n)
    f(n) = g(n) + h(n)
    WHERE
        g(n) = cumulative path cost
        h(n) = heuristic cost
    
    '''

    '''
    need to use default heuristic somewhere?
    Heuristic functions already written, just use them
    '''

    # uncomment line below to see tests run one at a time, helpful in conjuction with debug statements
    #input("hit enter to continue to next test")

    # set display to True to see debug statements, helpful for seeing step by step
    display = False

    # Keep track of parent child links
    child_to_parent_dict = {}

    start_state = problem.getStartState()

    # key is child node and value is (parent, action)
    child_to_parent_dict[start_state] = (None, None)

    # keep track of costs to get to a state i.e. g(n)
    shortest_cost_to_state = {start_state: 0}
    # keep track of heuristic cost to state i.e. h(n)
    heuristic_cost_to_state = {start_state: heuristic(start_state, problem)}

    calculated_priority = shortest_cost_to_state[start_state] +\
                          heuristic_cost_to_state[start_state]

    #keep track of f(n)
    priority_of_states = {start_state: calculated_priority}

    # initialize prio queue
    frontier = util.PriorityQueue()

    #heuristic takes a state as first param and problem as second
    #third element is (path cost, heuristic cost)

    frontier.push(start_state, calculated_priority)
    # heap shows (priority, count, state)

    # initizalize explored
    explored = []

    if display:
        print("\nNEW TEST\n")

    while not frontier.isEmpty():
        #current node is parent node
        #this only returns the item or "state" not anything else
        current_state = frontier.pop()

        #reached goal!
        if problem.isGoalState(current_state):
            if display:
                print("Goal state has been reached")
            break

        #if we already explored the state, skip it
        if current_state in explored:
            if display:
                print(f"Current State: {current_state} has already been explored")
            continue

        # add to explored
        explored.append(current_state)

        parent_state = current_state

        if display:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            f_list = frontier.heap
            print(f"Current Frontier: {f_list}")
            print(f"Current State is {current_state}")

        counter = 0

        # look through all successor NODES
        for each_child_node in problem.getSuccessors(parent_state):
            child_state = each_child_node[0]

            if child_state in explored:
                if display:
                    print(f"Child State: {child_state} already in explored, going to next node")
                    print("-----------------------------------------------------------------------")
                continue

            direction = each_child_node[1]
            cost_to_child_node_from_parent = each_child_node[2]
            # g(n) -> cumulative path cost
            cumulative_cost_to_child_node = shortest_cost_to_state[parent_state] + cost_to_child_node_from_parent
            # h(n) -> heuristic cost only
            heuristic_cost_of_child_state = heuristic(child_state, problem)

            #update heuristic cost tracker
            heuristic_cost_to_state[child_state] = heuristic_cost_of_child_state
            #determine priority -> f(n) = g(n) + h(n)
            calculated_priority = cumulative_cost_to_child_node + heuristic_cost_of_child_state

            if display:
                counter += 1
                print("----------------------------------------------------------------------------------------------")
                print(f"From {parent_state} to {child_state} go via {direction} at a cost of {cumulative_cost_to_child_node}")
                print(f"hueristic cost = {heuristic_cost_of_child_state}")
                print(f"Priority Value = {cumulative_cost_to_child_node} + {heuristic_cost_of_child_state} = {calculated_priority}")

            frontier_list = frontier.heap
            # check if the child is already in frontier
            if isAlreadyInHeap(frontier_list, child_state):
                if display:
                    print(f"{child_state} is already in frontier with a priority of {priority_of_states[child_state]} "
                          f"and current calc priority is {calculated_priority}")

                # if the child has a better priority i.e. shorter path found to child state
                if hasBetterPrio(frontier_list, child_state, calculated_priority):

                    if display:
                        print("UPDATING heap with current priority, and child to parent, and shortest cost to child ")

                    frontier.update(child_state, calculated_priority)
                    # store cost to child node
                    shortest_cost_to_state[child_state] = cumulative_cost_to_child_node

                    #store priority
                    priority_of_states[child_state] = calculated_priority

                    # store path from parent to child, key:child value:parent
                    child_to_parent_dict[child_state] = (parent_state, direction)

                    if display:
                        print(
                                "----------------------------------------------------------------------------------------------")

                else:
                    continue
            else:
                if display:
                    print(f"child state: {child_state} not in heap, adding {child_state} to heap")
                #if not in frontier then push with cumulative cost
                frontier.push(child_state,calculated_priority)
                #store cost to child node
                shortest_cost_to_state[child_state] = cumulative_cost_to_child_node
                # store priority
                priority_of_states[child_state] = calculated_priority
                #store path from parent to child, key:child value:parent
                child_to_parent_dict[child_state] = (parent_state, direction)

        if display:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    if display:
        print("A* Complete")

    return getUCSPath(child_to_parent_dict,current_state, start_state)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
