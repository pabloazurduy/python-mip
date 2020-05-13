from mip import Model
import numpy as np 
from mip.constants import CONTINUOUS
import logging

logger = logging.getLogger(__name__)

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

    model_copy =  model.copy()
    for inc_crt in model.constrs:
        logger.debug('removing temporally {}'.format(inc_crt.name))
        aux_model_inc_crt_idx = [crt.name for crt in model_copy.constrs].index(inc_crt.name)
        aux_model_inc_crt = model_copy.constrs[aux_model_inc_crt_idx]
        logger.debug('removing temporally {}'.format(aux_model_inc_crt.name))
        model_copy.remove(aux_model_inc_crt)