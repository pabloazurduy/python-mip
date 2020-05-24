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

# logger = logging.getLogger(__name__)
logger = logging.getLogger('conflict')

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
        if method == "additive-algorithm":
            return self.additive_algorithm(model)
            
            
    def deletion_filter(self, 
                        model:"Model")-> ConstrList:
        
        # 1. create a model with all constraints but one 
        aux_model = model.copy()
        aux_model.objective = 1
        aux_model.emphasis = 1 # feasibility
        aux_model.preprocess = 1 # -1  automatic, 0  off, 1  on.

        logger.debug('starting deletion_filter algorithm')
        
        for inc_crt in model.constrs:
            aux_model_inc_crt_idx = [crt.name for crt in aux_model.constrs].index(inc_crt.name)
            aux_model_inc_crt = aux_model.constrs[aux_model_inc_crt_idx]
            aux_model.remove(aux_model_inc_crt) # temporally remove inc_crt  
            
            aux_model.optimize() 
            status = aux_model.status
            # 2. test feasibility, if feasible, return dropped constraint to the set 
            # 2.1 else removed it permanently 
            # logger.debug('status {}'.format(status))
            if status == OptimizationStatus.INFEASIBLE:
                logger.debug('removing permanently {}'.format(inc_crt.name))
                continue
            elif status in [OptimizationStatus.FEASIBLE,OptimizationStatus.OPTIMAL] :
                aux_model.add_constr(inc_crt.expr, name= inc_crt.name)

        iis = aux_model.constrs

        return iis 


    def additive_algorithm(self,
                           model:"Model")-> ConstrList:
        
        # Create some aux models to test feasibility of the set of constraints  
        aux_model_testing =  Model()
        for var in model.vars:
            aux_model_testing.add_var(name=var.name, 
                                      lb=var.lb,
                                      ub= var.ub, 
                                      var_type=var.var_type,
                                      # obj= var.obj, 
                                      # column=var.column   #!! libc++abi.dylib: terminating with uncaught exception of type CoinError
                                      )
        aux_model_testing.objective = 1
        aux_model_testing.emphasis = 1 # feasibility
        aux_model_testing.preprocess = 1 # -1  automatic, 0  off, 1  on.
        aux_model_iis = aux_model_testing.copy() # a second aux model to test feasibility of the incumbent iis
        
        # algorithm start 
        all_constraints = model.constrs
        testing_crt_set = ConstrList(model=aux_model_testing) #T
        iis = ConstrList(model=aux_model_iis) #I
        
        while True:
            for crt in all_constraints:
                testing_crt_set.add(crt.expr, name=crt.name)
                aux_model_testing.constrs = testing_crt_set
                aux_model_testing.optimize()

                if aux_model_testing.status ==  OptimizationStatus.INFEASIBLE:
                    iis.add(crt.expr, name=crt.name)
                    aux_model_iis.constrs = iis
                    aux_model_iis.optimize()

                    if aux_model_iis.status == OptimizationStatus.INFEASIBLE:
                        return iis  
                    elif aux_model_iis.status in [OptimizationStatus.FEASIBLE,OptimizationStatus.OPTIMAL] :
                        testing_crt_set = ConstrList(model=aux_model_testing)
                        for crt in iis: # basically this loop is for set T=I // aux_model_iis =  iis.copy() 
                            testing_crt_set.add(crt.expr, name=crt.name)
                        break     
            
        

def build_infeasible_cont_model(num_constraints:int = 10, num_infeasible_sets:int = 20) -> Model:
    # build an infeasible model, based on many redundant constraints 
    mdl = Model(name='infeasible_model_continuous')
    var = mdl.add_var(name='var', var_type=CONTINUOUS, lb=-1000, ub=1000)
    
    
    for idx,rand_constraint in enumerate(np.linspace(1,1000,num_constraints)):
        crt = mdl.add_constr(var>=rand_constraint, name='lower_bound_{}'.format(idx))
        logger.debug('added {} to the model'.format(crt))
    
    num_constraint_inf = int(num_infeasible_sets/num_constraints)
    for idx,rand_constraint in enumerate(np.linspace(-1000,-1,num_constraint_inf)):
        crt = mdl.add_constr(var<=rand_constraint, name='upper_bound_{}'.format(idx))
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
    iis = cf.find_iis(model, "additive-algorithm")
    logger.debug([crt.name for crt in iis])

