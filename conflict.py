import logging
from copy import copy, deepcopy
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

logger = logging.getLogger(__name__)
class ConflictFinder:

    def __init__(self):
        pass

    def find_iis(self,
                 model: "Model", 
                 method: str = "deletion-filter",) -> ConstrList:
        # check if infeasible 
        assert model.status == OptimizationStatus.INFEASIBLE, "model is not infeasible"
        # assert ,is not because time limit 
        if method == "deletion-filter":
            return self.deletion_filter(model)
            
    def deletion_filter(self, 
                        model:"Model")-> ConstrList:
    
        # 1. create a model with all constraints but one 
        aux_model = model.copy()
        aux_model.objective = 1
        crt_all = copy(aux_model.constrs)

        aux_model.emphasis = 1 # feasibility
        aux_model.preprocess = 1 # -1  automatic, 0  off, 1  on.
        logger.debug('starting deletion_filter algorithm')
        
        for inc_crt in crt_all:
            aux_model.constrs.remove([inc_crt]) # temporally remove inc_crt 
            aux_model.optimize() 
            status = aux_model.status
            # 2. test feasibility, if feasible, return dropped constraint to the set 
            # 2.1 else removed it permanently 
            if status == OptimizationStatus.INFEASIBLE:
                logger.debug('removing permanently {}'.format(inc_crt))
                continue
            elif status == OptimizationStatus.FEASIBLE:
                aux_model.constrs.add([inc_crt])
        iis = aux_model.constrs
        return iis  

def build_infeasible_cont_model(num_constraints:int = 10, num_infeasible_sets:int = 20) -> Model:
    # build an infeasible model, based on many redundant constraints 
    mdl = Model(name='infeasible_model_continuous')
    var = mdl.add_var(name='var', var_type=CONTINUOUS, lb=-1000, ub=1000)
    
    
    for idx,rand_constraint in enumerate(np.linspace(1,1000,num_constraints)):
        crt = mdl.add_constr(var>=rand_constraint, name='lower_bound_{}'.format(idx))
        logger.debug('added {} to the model'.format(crt))
    
    num_constraint_inf = int(num_infeasible_sets/num_constraints)
    for idx,rand_constraint in enumerate(np.linspace(-1000,-1,num_constraint_inf)):
        crt = mdl.add_constr(var<=rand_constraint, name='lower_bound_{}'.format(idx))
        logger.debug('added {} to the model'.format(crt))
        
    mdl.emphasis = 1 # feasibility
    mdl.preprocess = 1 # -1  automatic, 0  off, 1  on.
    # mdl.pump_passes TODO configure to feasibility emphasis 
    return mdl


import sys
if __name__ == "__main__":
    
    # logger config
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # model experiment
    model = build_infeasible_cont_model()
    logger.debug(model.status)
    model.optimize()
    logger.debug(model.status)

    cf = ConflictFinder()
    iss = cf.find_iis(model)
    logger.debug(iis)
