import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from mip.constants import (
    INF,
    GUROBI,
    CBC,
    CONTINUOUS,
    OptimizationStatus,
    SearchEmphasis,
    VERSION,
    BINARY,
    INTEGER
)
from mip.model import Model
from mip.lists import ConstrList

from mip.exceptions import (
    InvalidLinExpr,
    InfeasibleSolution,
    SolutionNotAvailable,
)
import numpy as np

class ConflictFinder:

    def __init__():
        # TODO
        pass

    def find_iis(self,
                 model: "Model", 
                 method: str = "deletion-filter",) -> ConstrList:
        # check if infeasible 
        assert model.status == InfeasibleSolution, "model is not infeasible"
        # assert ,is not because time limit 
        model.emphasis = 1 # (FEASIBILITY)
        if method == "deletion-filter":
            return self.deletion_filter(model)
            
    def deletion_filter(self, 
                        model:"Model")-> ConstrList:
        # get all constraints
        constraint_list = model.constrs
        # 1. create a model with all constraints but one 


        # 2. test feasibility, if feasible, return dropped constraint to the set 
        # 2.1 else removed it permanently 

        pass


import sys
if __name__ == "__main__":

    logger = logging.getLogger('conflict_application')
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # build an infeasible model, based on many redundant constraints 
    mdl = Model(name='infeasible_model_continuous')
    var = mdl.add_var(name='var', var_type=CONTINUOUS, lb=-1000, ub=1000)
    num_constraints = 10
    num_infeasible_sets = 20
    for idx,rand_constraint in enumerate(np.linspace(0,1000,num_constraints)):
        mdl.add_constr(var>=rand_constraint, name='lower_bound_{}'.format(idx))
        logger.debug('creating an instance of Auxiliary')
    
    num_constraint_inf = int(num_infeasible_sets/num_constraints)
    for idx,rand_constraint in enumerate(np.linspace(-1000,0,num_constraint_inf)):
        mdl.add_constr(var<=rand_constraint, name='lower_bound_{}'.format(idx))
        logger.debug('creating an instance of Auxiliary')
        
    mdl.emphasis = 1 # feasibility
    mdl.preprocess = 1 # -1  automatic, 0  off, 1  on.
    # mdl.pump_passes TODO configure to feasibility emphasis 
    mdl.optimize()
