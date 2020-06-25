import logging
import sys
from copy import copy, deepcopy
from enum import Enum
from functools import total_ordering, reduce
from typing import Any, Dict, List, Optional, Tuple, Union
import random
from collections import Counter
from operator import add

import mip
import numpy as np

# logger = logging.getLogger(__name__)
logger = logging.getLogger('conflict')

class ConflictFinder:
    """ This class groups some IIS (Irreducible Infeasible Set) search algorithms 
    """    
    def __init__(self):
        pass

    def find_iis(self,
                 model: "mip.Model", 
                 method: str = "deletion-filter",) -> mip.ConstrList:
        """ main method to find an IIS, this method is just a grouping of the other implementations 

        Args:
            model (mip.Model): Infeasible model to search
            method (str, optional): [description]. Defaults to "deletion-filter".

        Returns:
            mip.ConstrList: [description]
        """               
        # check if infeasible 
        assert model.status == mip.OptimizationStatus.INFEASIBLE, 'model is not infeasible'
        # assert ,is not because time limit 
        if method == "deletion-filter":
            return self.deletion_filter(model)
        if method == "additive-algorithm":
            return self.additive_algorithm(model)        
            
    def deletion_filter(self, 
                        model:"mip.Model")-> mip.ConstrList:
        
        # 1. create a model with all constraints but one 
        aux_model = model.copy()
        aux_model.objective = 1
        aux_model.emphasis = 1 # feasibility
        aux_model.preprocess = 1 # -1  automatic, 0  off, 1  on.

        logger.debug('starting deletion_filter algorithm')
        
        for inc_crt in model.constrs:
            aux_model_inc_crt = aux_model.constr_by_name(inc_crt.name) # find constraint by name 
            aux_model.remove(aux_model_inc_crt) # temporally remove inc_crt  
            
            aux_model.optimize() 
            status = aux_model.status
            # 2. test feasibility, if feasible, return dropped constraint to the set 
            # 2.1 else removed it permanently 
            # logger.debug('status {}'.format(status))
            if status == mip.OptimizationStatus.INFEASIBLE:
                logger.debug('removing permanently {}'.format(inc_crt.name))
                continue
            elif status in [mip.OptimizationStatus.FEASIBLE, mip.OptimizationStatus.OPTIMAL] :
                aux_model.add_constr(inc_crt.expr, name= inc_crt.name)

        iis = aux_model.constrs

        return iis 


    def additive_algorithm(self,
                           model:"mip.Model")-> mip.ConstrList:
        
        # Create some aux models to test feasibility of the set of constraints  
        aux_model_testing =  mip.Model()
        for var in model.vars:
            aux_model_testing.add_var(name = var.name, 
                                      lb = var.lb,
                                      ub = var.ub, 
                                      var_type = var.var_type,
                                      # obj= var.obj, 
                                      # column=var.column   #!! libc++abi.dylib: terminating with uncaught exception of type CoinError
                                      )
        aux_model_testing.objective = 1
        aux_model_testing.emphasis = 1 # feasibility
        aux_model_testing.preprocess = 1 # -1  automatic, 0  off, 1  on.
        aux_model_iis = aux_model_testing.copy() # a second aux model to test feasibility of the incumbent iis
        
        # algorithm start 
        all_constraints = model.constrs
        testing_crt_set = mip.ConstrList(model=aux_model_testing) #T
        iis = mip.ConstrList(model=aux_model_iis) #I
        
        while True:
            for crt in all_constraints:
                testing_crt_set.add(crt.expr, name=crt.name)
                aux_model_testing.constrs = testing_crt_set
                aux_model_testing.optimize()

                if aux_model_testing.status ==  mip.OptimizationStatus.INFEASIBLE:
                    iis.add(crt.expr, name=crt.name)
                    aux_model_iis.constrs = iis
                    aux_model_iis.optimize()

                    if aux_model_iis.status == mip.OptimizationStatus.INFEASIBLE:
                        return iis  
                    elif aux_model_iis.status in [mip.OptimizationStatus.FEASIBLE,mip.OptimizationStatus.OPTIMAL] :
                        testing_crt_set = mip.ConstrList(model=aux_model_testing)
                        for crt in iis: # basically this loop is for set T=I // aux_model_iis =  iis.copy() 
                            testing_crt_set.add(crt.expr, name=crt.name)
                        break     
            
    def deletion_filter_milp_ir_lc_bd(self,
                                      model:"mip.Model")-> mip.ConstrList:    
        
        raise NotImplementedError('WIP')
        # major constraint sets definition  
        t_aux_model = mip.Model(name='t_auxiliary_model')
        iis_aux_model = mip.Model(name='t_auxiliary_model')

        linear_constraints = mip.ConstrList(model = t_aux_model) # all the linear model constraints
        variable_bound_constraints = mip.ConstrList(model = t_aux_model) # all the linear model constrants related specifically for the variable bounds
        integer_varlist_crt =  mip.VarList(model = t_aux_model) # the nature vars constraints for vartype in Integer/Binary 
        
        # fill the above sets with the constraints
        for crt in model.constrs:
            linear_constraints.add(crt.expr, name=crt.name)
        for var in model.vars:
            if var.lb != - mip.INF:
                variable_bound_constraints.add(var >= var.lb ,name = '{}_lb_crt'.format(var.name))
            if var.ub != mip.INF:
                variable_bound_constraints.add(var <= var.ub ,name = '{}_ub_crt'.format(var.name))
        for var in model.vars:
            if var.var_type in (mip.INTEGER, mip.BINARY):
                integer_varlist_crt.add(var)
        
        status = 'IIS'
        # add all LC,BD to the incumbent, T= LC + BD
        for var in model.vars: # add all variables as if they where CONTINUOUS and without bonds (because this will be separated)              
            iis_aux_model.add_var(name=var.name, 
                                  lb = -mip.INF,
                                  ub = mip.INF, 
                                  var_type=mip.CONTINUOUS
                                 )
        for crt in  linear_constraints + variable_bound_constraints:
            iis_aux_model.add_constr(crt.expr, name=crt.name)
         
        iis_aux_model.optimize()
        if iis_aux_model.status ==  mip.OptimizationStatus.INFEASIBLE:
            # if infeasible means that this is a particular version of an LP
            return(self.deletion_filter(model)) # (STEP 2)
        
        # add all the integer constraints to the model
        iis_aux_model.vars.remove([var for var in integer_varlist_crt]) # remove all integer variables
        for var in integer_varlist_crt:
            iis_aux_model.add_var(name=var.name, 
                                  lb = -mip.INF,
                                  ub = mip.INF, 
                                  var_type=var.var_type # this will add the var with his original type
                                 )   
        # filter IR constraints that create infeasibility (STEP 1)
        for var in integer_varlist_crt:
            iis_aux_model.vars.remove(iis_aux_model.var_by_name(var.name))
            iis_aux_model.add_var(name=var.name, 
                        lb = -mip.INF,
                        ub = mip.INF, 
                        var_type=mip.CONTINUOUS # relax the integer constraint over var 
                        )   
            iis_aux_model.optimize()
            # if infeasible then update incumbent T = T-{ir_var_crt}
            # else continue 
        # STEP 2 filter lc constraints 
        # STEP 3 filter BD constraints 
        # return IS o IIS
         

    def deletion_filter_milp_lc_ir_bd(self,
                                      model:"mip.Model")-> mip.ConstrList:    
        # TODO
        raise NotImplementedError

class ConstraintPriority(Enum):
    # constraints levels
    VERY_LOW_PRIORITY = 1
    LOW_PRIORITY = 2
    NORMAL_PRIORITY = 3
    MID_PRIORITY = 4
    HIGH_PRIORITY = 5
    VERY_HIGH_PRIORITY = 6
    MANDATORY = 7

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class ConflictResolver():
    
    # mapper for constraint naming (while the attribute 'crt_importance' not in the mip.Constraint class)
    PRIORITY_MAPPER = {
        '_l1':ConstraintPriority.VERY_LOW_PRIORITY,
        '_l2':ConstraintPriority.LOW_PRIORITY,
        '_l3':ConstraintPriority.NORMAL_PRIORITY,
        '_l4':ConstraintPriority.MID_PRIORITY,
        '_l5':ConstraintPriority.HIGH_PRIORITY,
        '_l6':ConstraintPriority.VERY_HIGH_PRIORITY,
        '_l7':ConstraintPriority.MANDATORY
    }

    def __init__(self):
        self.iis_num_iterations = 0
        self.iis_iterations = []
        self.relax_slack_iterations = []
    
    
    @property
    def slack_by_crt(self) -> dict:
        return dict(reduce(add, (Counter(dict(x)) for x in self.relax_slack_iterations)))

    def hierarchy_relaxer(self, 
                          model:mip.Model, 
                          relaxer_objective:str = 'min_abs_slack_val', 
                          default_priority:ConstraintPriority = ConstraintPriority.MANDATORY ) -> mip.Model:
        """[summary]

        Args:
            model (mip.Model): [description]
            relaxer_objective (str, optional): [description]. Defaults to 'min_abs_slack_val'.
            default_priority (ConstraintPriority, optional): [description]. Defaults to ConstraintPriority.MANDATORY.

        Raises:
            Exception: [description]

        Returns:
            mip.Model: [description]
        """        

        # check if infeasible 
        assert model.status == mip.OptimizationStatus.INFEASIBLE, 'model is not infeasible'

        relaxed_model = model.copy()  
        relaxed_model._status = model._status #TODO solve this in a different way
        # 0 map priorities 
        crt_priority_dict = self.map_constraint_priorities(model, default_priority = default_priority)

        cf = ConflictFinder()
        while True:
            # 1. find iis 
            iis = cf.find_iis(relaxed_model, 'deletion-filter')
            self.iis_iterations.append([crt.name for crt in iis]) # track iteration 
            self.iis_num_iterations+=1 # track iteration 

            iis_priority_mapping = { crt.name :crt_priority_dict[crt.name] for crt in iis}
            # check if "relaxable" model mapping 
            if set(iis_priority_mapping.values()) == set([ConstraintPriority.MANDATORY]):
                raise Exception('Infeasible model, is not possible to relax MANDATORY constraints')
            
            # 2. relax iss 
            slack_dict = self.relax_iss(iis, iis_priority_mapping, relaxer_objective = relaxer_objective)

            self.relax_slack_iterations.append(slack_dict)
            # 3. add the slack variables to the original problem 
            relaxed_model = self.relax_constraints(relaxed_model, slack_dict)

            # 4. check if feasible 
            relaxed_model.emphasis = 1 # feasibility
            relaxed_model.optimize()
            if relaxed_model.status in [mip.OptimizationStatus.FEASIBLE, mip.OptimizationStatus.OPTIMAL]:
                logger.debug('finished relaxation process !')
                break 
            else:
                logger.debug('relaxed the current IIS, still infeasible, searching for a new IIS to relax')
                logger.debug('relaxed constraints {0}'.format(list(slack_dict.keys())))
                
    
    @classmethod
    def map_constraint_priorities(cls, model:mip.Model, mapper:dict = PRIORITY_MAPPER, default_priority:ConstraintPriority = ConstraintPriority.MANDATORY ) -> dict:
        
        crt_importance_dict = {} # dict with name
        crt_name_list = [crt.name for crt in model.constrs]
        
        #check unique names 
        assert len(crt_name_list) == len(set(crt_name_list)), 'names in constraints must be unique to use conflict refiner, please rename them'

        # TODO: this could be optimized
        for crt_name in crt_name_list:
            for key in mapper.keys():
                if key in crt_name:
                    crt_importance_dict[crt_name] = mapper[key]
                    break
            
        non_defined_crt = [crt_name for crt_name in crt_name_list if crt_name not in  crt_importance_dict.keys()]
        for crt_name in non_defined_crt:
            crt_importance_dict[crt_name] = default_priority

        return crt_importance_dict

    @classmethod 
    def relax_iss(cls, 
                  iis:mip.ConstrList, 
                  iis_priority_mapping:dict, 
                  relaxer_objective:str = 'min_abs_slack_val', 
                  big_m = 10e8) -> dict:

        """[summary]

        Args:
            iis (mip.ConstrList): [description]
            iis_priority_mapping (dict): [description]
            relaxer_objective (str, optional): [description]. Defaults to 'min_abs_slack_val'.
            big_m ([type], optional): [description]. Defaults to 10e10.

        Returns:
            dict: [description]
        """        
        relax_iss_model = mip.Model()
        lowest_priority =  min(list(iis_priority_mapping.values()))
        to_relax_crts = [crt for crt in iis if iis_priority_mapping[crt.name] == lowest_priority]

        # create a model that only contains the iis
        slack_vars = {}
        abs_slack_vars = {}
        abs_slack_cod_vars = {}
        for crt in iis:
            for var in crt._Constr__model.vars:
                relax_iss_model.add_var(name=var.name, 
                                        lb = var.lb,
                                        ub = var.ub, 
                                        var_type=var.var_type,
                                        obj = var.obj
                                        )
            if crt in to_relax_crts:
                # if this is a -toberelax- constraint 
                slack_vars[crt.name] = relax_iss_model.add_var( name = '{0}__{1}'.format(crt.name, 'slack'), 
                                                                lb = -mip.INF, 
                                                                ub = mip.INF, 
                                                                var_type=mip.CONTINUOUS)
            
                abs_slack_vars[crt.name] = relax_iss_model.add_var( name = '{0}_abs'.format(slack_vars[crt.name].name), 
                                                                    lb = 0, 
                                                                    ub = mip.INF, 
                                                                    var_type=mip.CONTINUOUS)

                abs_slack_cod_vars[crt.name] = relax_iss_model.add_var( name = '{0}_abs_cod'.format(slack_vars[crt.name].name), 
                                                                        var_type=mip.BINARY)
                                                            
                # add relaxed constraint to model 
                relax_expr =  crt.expr + slack_vars[crt.name]
                relax_iss_model.add_constr(relax_expr, name='{}_relaxed'.format(crt.name))

                # add abs(slack) variable encoding constraints
                relax_iss_model.add_constr(abs_slack_vars[crt.name] >= slack_vars[crt.name] , name='{}_positive_min_bound'.format(slack_vars[crt.name].name))
                relax_iss_model.add_constr(abs_slack_vars[crt.name] >= -slack_vars[crt.name] , name='{}_negative_min_bound'.format(slack_vars[crt.name].name))
                
                relax_iss_model.add_constr(abs_slack_vars[crt.name] <= slack_vars[crt.name] + big_m* abs_slack_cod_vars[crt.name] , 
                                            name='{}_positive_max_bound'.format(slack_vars[crt.name].name))
                
                relax_iss_model.add_constr(abs_slack_vars[crt.name] <= -slack_vars[crt.name] + big_m* (1-abs_slack_cod_vars[crt.name])
                                           ,name='{}_negative_max_bound'.format(slack_vars[crt.name].name))                


            else:
                # if not to be relaxed we added directly to the model
                relax_iss_model.add_constr(crt.expr, name='{}_original'.format(crt.name))
                

        
        # find the min abs value of the slack variables 
        relax_iss_model.objective = mip.xsum(list(abs_slack_vars.values()))
        relax_iss_model.sense = mip.MINIMIZE
        relax_iss_model.optimize()
        if relax_iss_model.status == mip.OptimizationStatus.INFEASIBLE:
            raise ValueError('relaxation model infeasible, usually is a problem with the big_m parameter')
            
        slack_dict = {}
        for crt in to_relax_crts:
            slack_dict[crt.name] = slack_vars[crt.name].x

        return slack_dict
    
    @classmethod
    def relax_constraints(cls, relaxed_model:mip.Model, slack_dict:dict) ->mip.Model:
        for crt_name in slack_dict.keys():
            crt_original = relaxed_model.constr_by_name(crt_name)

            relax_expr =  crt_original.expr + slack_dict[crt_name]
            relaxed_model.add_constr(relax_expr, name=crt_original.name)
            relaxed_model.remove(crt_original) # remove constraint 

        return relaxed_model


def build_infeasible_cont_model(num_constraints:int = 10, 
                                num_infeasible_sets:int = 20) -> mip.Model:
    # build an infeasible model, based on many redundant constraints 
    mdl = mip.Model(name='infeasible_model_continuous')
    var = mdl.add_var(name='x', var_type=mip.CONTINUOUS, lb=-1000, ub=1000)
    
    
    for idx,rand_constraint in enumerate(np.linspace(1,1000,num_constraints)):
        crt = mdl.add_constr(var>=rand_constraint, name='lower_bound_{0}_l{1}'.format(idx,random.randint(1,6) ))
        logger.debug('added {} to the model'.format(crt))
    
    num_constraint_inf = int(num_infeasible_sets/num_constraints)
    for idx,rand_constraint in enumerate(np.linspace(-1000,-1,num_constraint_inf)):
        crt = mdl.add_constr(var<=rand_constraint, name='upper_bound_{0}_l{1}'.format(idx, random.randint(1,7)))
        logger.debug('added {} to the model'.format(crt))
        
    mdl.emphasis = 1 # feasibility
    mdl.preprocess = 1 # -1  automatic, 0  off, 1  on.
    # mdl.pump_passes TODO configure to feasibility emphasis 
    return mdl


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

    cr = ConflictResolver()
    cr.hierarchy_relaxer(model)
    print(cr.slack_by_crt)