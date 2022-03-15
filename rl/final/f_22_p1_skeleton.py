from rl.distribution import Distribution, Constant, Gaussian, Choose, SampledDistribution
from itertools import product
from collections import defaultdict
import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Dict, Iterable, Generic, Set

import numpy as np

from rl.distribution import Categorical, Choose
from rl.markov_process import NonTerminal, State, Terminal
from rl.markov_decision_process import (MarkovDecisionProcess, FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf

import random 

from dataclasses import dataclass
from rl import dynamic_programming

from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.td import PolicyFromQType, epsilon_greedy_action


S = TypeVar('S')
A = TypeVar('A')


class TabularQValueFunctionApprox(Generic[S, A]):
    '''
    A basic implementation of a tabular function approximation with constant learning rate of 0.1
    also tracks the number of updates per state
    You should use this class in your implementation
    '''

    
    def __init__(self):
        self.counts: Mapping[Tuple[NonTerminal[S], A], int] = defaultdict(int)
        self.values: Mapping[Tuple[NonTerminal[S], A], float] = defaultdict(float)
    
    def update(self, k: Tuple[NonTerminal[S], A], tgt):
        alpha = 0.1
        self.values[k] = (1 - alpha) * self.values[k] + tgt * alpha
        self.counts[k] += 1
    
    def __call__(self, x_value: Tuple[NonTerminal[S], A]) -> float:
        return self.values[x_value]

    
def greedy_double_q(
    q1: TabularQValueFunctionApprox[S,A],
    q2: TabularQValueFunctionApprox[S,A],
    nt_state: NonTerminal[S],
    actions: Set[A],
    epsilon: float
) -> A:
    greedy_action: A = max(
        ((a, q1((nt_state, a)) + q2((nt_state,a))) for a in actions),
        key=operator.itemgetter(1)
    )[0]
    return Categorical(
        {a: epsilon / len(actions) +
         (1 - epsilon if a == greedy_action else 0.) for a in actions}
    ).sample()



def double_q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    gamma: float
) -> Iterator[TabularQValueFunctionApprox[S, A]]:

    ##### Your Code HERE #########
    q1 = TabularQValueFunctionApprox[S,A]()
    q2 = TabularQValueFunctionApprox[S,A]()
    q_approx = TabularQValueFunctionApprox[S,A]()
    yield q_approx
    while True:
        state: NonTerminal[S] = states.sample()
        t: int = 0

        while isinstance(state, NonTerminal):
            action: A = greedy_double_q(
                q1=q1,
                q2=q2,
                nt_state=state,
                actions=set(mdp.actions(state)),
                epsilon=0.1
            )
            next_state, reward = mdp.step(state, action).sample()

            unif_var = random.uniform(0,1)

            if unif_var > 0.5:
                next_return: float = 0
                if isinstance(next_state, NonTerminal):
                    max_act: A = max(
                        ((a, q2((next_state, a))) for a in mdp.actions(next_state)),   
                        key=operator.itemgetter(1)
                        )[0]
                    next_return = q1((next_state,max_act))
                q1.update(k= ((state, action)), tgt= reward + gamma*next_return)
            else:
                next_return: float = 0
                if isinstance(next_state, NonTerminal):
                    max_act: A = max(
                        ((a, q1((next_state, a))) for a in mdp.actions(next_state)),
                        key=operator.itemgetter(1)
                    )[0]
                    next_return = q2((next_state,max_act))
                q2.update(k= ((state, action)), tgt= reward + gamma*next_return)
            q_approx.values[(state,action)] = (q1((state,action)) + q1((state,action)))/2
            q_approx.counts[(state,action)] += 1
            t += 1
            state = next_state
        yield q_approx
    ##### End Your Code HERE #########
    
            
def q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    gamma: float
) -> Iterator[TabularQValueFunctionApprox[S, A]]:

    ##### Your Code HERE #########
    qfunc = TabularQValueFunctionApprox[S,A]()
    yield qfunc
    while True:
        state: NonTerminal[S] = states.sample()
        t: int = 0
        while isinstance(state, NonTerminal):
            action: A = epsilon_greedy_action(
                q=qfunc,
                nt_state=state,
                actions=set(mdp.actions(state)),
                ϵ=0.1
            )
            next_state, reward = mdp.step(state, action).sample()
            next_return: float = max(
                qfunc((next_state, a))
                for a in mdp.actions(next_state)
            ) if isinstance(next_state, NonTerminal) else 0.
            qfunc.update(k= ((state, action)), tgt= reward + gamma*next_return)
            t += 1
            state = next_state
        yield qfunc
    ##### End Your Code HERE #########



@dataclass(frozen=True)
class P1State:
    '''
    Add any data and functionality you need from your state
    '''
    ##### Your Code HERE #########
    value: str
    ##### End Your Code HERE #########
    

class P1MDP(MarkovDecisionProcess[P1State, str]):
    
    def __init__(self, n):
        self.n = n
        
    def actions(self, state: NonTerminal[P1State]) -> Iterable[str]:
        '''
        return the actions available from: state
        '''
        ##### Your Code HERE #########
        if state.state.value == "A":
            return ["a1", "a2"]
        else:
            blist = []
            for i in range(self.n):
                blist.append(str(i))
            return blist

        ##### End Your Code HERE #########
    
    def step(
        self,
        state: NonTerminal[P1State],
        action: str
    ) -> Distribution[Tuple[State[P1State], float]]:
        '''
        return the distribution of next states conditioned on: (state, action)
        '''
        ##### Your Code HERE #########
        if state.state.value == "A":
            if action == "a1":
                return Categorical({((NonTerminal(P1State("B")), 0)): 1})
            else:
                return Categorical({((Terminal(P1State("Done")), 0)): 1})
        else:
            rando = np.random.normal(-0.1,1)
            return Categorical({((Terminal(P1State("Done")), rando)): 1})
        ##### End Your Code HERE #########
