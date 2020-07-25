import logging
import sys
from collections import Counter
from copy import copy, deepcopy
from enum import Enum
from functools import reduce, total_ordering
from operator import add
from typing import Any, Dict, List, Optional, Tuple, Union

import mip
import numpy as np

# logger = logging.getLogger(__name__)
logger = logging.getLogger('conflict')

class ConflictFinder:
    """ This class groups some IIS (Irreducible Infeasible Set) search algorithms 
    """    
    def __init__(self, model:mip.Model):
        assert model.status == mip.OptimizationStatus.INFEASIBLE, 'model is not infeasible'
        self.model = model 


    def find_iis(self,
                 method: str = "deletion-filter",) -> mip.ConstrList:
        """ main method to find an IIS, this method is just a grouping of the other implementations 

        Args:
            model (mip.Model): Infeasible model where to find the IIS
            method (str, optional): name of the method to use ["deletion-filter", "additive_algorithm"]. Defaults to 'deletion-filter". 

        Returns:
            mip.ConstrList: IIS constraint list 
        """               
        # check if infeasible 
        assert self.model.status == mip.OptimizationStatus.INFEASIBLE, 'model is not infeasible'
        # assert ,is not because time limit 
        if method == "deletion-filter":
            return self.deletion_filter()
        if method == "additive-algorithm":
            return self.additive_algorithm()        
            
    def deletion_filter(self)-> mip.ConstrList:
        """ deletion filter algorithm for search an IIS

        Args:
            model (mip.Model): Infeasible model 

        Returns:
            mip.ConstrList: IIS
        """        
        # 1. create a model with all constraints but one 
        aux_model = self.model.copy()
        aux_model.objective = 1
        aux_model.emphasis = 1 # feasibility
        aux_model.preprocess = 1 # -1  automatic, 0  off, 1  on.

        logger.debug('starting deletion_filter algorithm')
        
        for inc_crt in self.model.constrs:
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


    def additive_algorithm(self)-> mip.ConstrList:
        """Additive algorithm to find an IIS

        Returns:
            mip.ConstrList: IIS
        """        
        # Create some aux models to test feasibility of the set of constraints  
        aux_model_testing =  mip.Model()
        for var in self.model.vars:
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
        all_constraints = self.model.constrs
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
            
    def deletion_filter_milp_ir_lc_bd(self)-> mip.ConstrList:    
        """Integer deletion filter algorithm (milp_ir_lc_bd)

        Raises:
            NotImplementedError: [description]

        Returns:
            mip.ConstrList: [description]
        """        
        raise NotImplementedError('WIP')
        # major constraint sets definition  
        t_aux_model = mip.Model(name='t_auxiliary_model')
        iis_aux_model = mip.Model(name='t_auxiliary_model')

        linear_constraints = mip.ConstrList(model = t_aux_model) # all the linear model constraints
        variable_bound_constraints = mip.ConstrList(model = t_aux_model) # all the linear model constrants related specifically for the variable bounds
        integer_varlist_crt =  mip.VarList(model = t_aux_model) # the nature vars constraints for vartype in Integer/Binary 
        
        # fill the above sets with the constraints
        for crt in self.model.constrs:
            linear_constraints.add(crt.expr, name=crt.name)
        for var in self.model.vars:
            if var.lb != - mip.INF:
                variable_bound_constraints.add(var >= var.lb ,name = '{}_lb_crt'.format(var.name))
            if var.ub != mip.INF:
                variable_bound_constraints.add(var <= var.ub ,name = '{}_ub_crt'.format(var.name))
        for var in self.model.vars:
            if var.var_type in (mip.INTEGER, mip.BINARY):
                integer_varlist_crt.add(var)
        
        status = 'IIS'
        # add all LC,BD to the incumbent, T= LC + BD
        for var in self.model.vars: # add all variables as if they where CONTINUOUS and without bonds (because this will be separated)              
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
            return(self.deletion_filter()) # (STEP 2)
        
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
         

    def deletion_filter_milp_lc_ir_bd(self)-> mip.ConstrList:    
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
        '_l7':ConstraintPriority.MANDATORY # This level is never going to be relaxed
    }

    def __init__(self, model:mip.Model):
        # check if infeasible 
        assert model.status == mip.OptimizationStatus.INFEASIBLE, 'model is not infeasible'

        self.model = model
        self.iis_num_iterations = 0
        self.iis_iterations = []
        self.relax_slack_iterations = []
    
    
    @property
    def slack_by_crt(self) -> dict:
        return dict(reduce(add, (Counter(dict(x)) for x in self.relax_slack_iterations)))

    def hierarchy_relaxer(self, 
                          relaxer_objective:str = 'min_abs_slack_val', 
                          default_priority:ConstraintPriority = ConstraintPriority.MANDATORY,
                          priority_mapper:dict = PRIORITY_MAPPER ) -> mip.Model:
        """ hierarchy relaxer algorithm, it's gonna find a IIS and then relax it using the objective function defined (`relaxer_objective`) and then update the model 
        with the relaxed constraints. This process runs until there's not more IIS on the model. 

        Args:
            relaxer_objective (str, optional): objective function of the relaxer model (IIS relaxer model). Defaults to 'min_abs_slack_val'.
            default_priority (ConstraintPriority, optional): If a constraint does not have a supported substring priority in the name, it will assign a default priority.
                                                             Defaults to ConstraintPriority.MANDATORY.

        Raises:
            Exception: [description]

        Returns:
            mip.Model: relaxed model
        """        

        
        relaxed_model = self.model.copy()  
        relaxed_model._status = self.model._status #TODO solve this in a different way
        # 0 map priorities 
        crt_priority_dict = self.map_constraint_priorities(model = self.model, mapper =  priority_mapper,default_priority = default_priority)

        cf = ConflictFinder(relaxed_model)
        while True:
            # 1. find iis 
            iis = cf.find_iis('deletion-filter')
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
                
        return relaxed_model
        
    @classmethod
    def map_constraint_priorities(cls, 
                                  model:mip.Model, 
                                  mapper:dict = PRIORITY_MAPPER, 
                                  default_priority:ConstraintPriority = ConstraintPriority.MANDATORY ) -> dict:
        """ this method is used to map {constraint_name: ConstraintPriority} for each constraint in the model. This map uses the constraint name 
            and looks for the `_l?` regex to map it to a corresponding level (those levels are described on the mapper dict)

        Args:
            model (mip.Model): [description]
            mapper (dict, optional): [description]. Defaults to PRIORITY_MAPPER.
            default_priority (ConstraintPriority, optional): [description]. Defaults to ConstraintPriority.MANDATORY.

        Returns:
            dict: {constraint_name: ConstraintPriority} pars for all the model.constrs 
        """        
        
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
                  big_m = 10e7) -> dict:

        """ This function is the sub module that finds the optimum relaxation for an IIS, given a crt priority mapping and a objective function

        Args:
            iis (mip.ConstrList): IIS constraint list 
            iis_priority_mapping (dict): maps {crt_name: ConstraintPriority}
            relaxer_objective (str, optional): objective function to use when relaxing. Defaults to 'min_abs_slack_val'.
            big_m (float, optional): this is just for abs value codification. Defaults to 10e10.

        Returns:
            dict: a slack variable dictionary with the value of the {constraint_name:slack.value} pair to be added to each constraint in order to make the IIS feasible
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
            raise ValueError('sub relaxation model infeasible, this could mean that in the IIS the mandatory constraints are infeasible sometimes. Also could mean that the big_m parameter is overflowed ')
            
        slack_dict = {}
        for crt in to_relax_crts:
            slack_dict[crt.name] = slack_vars[crt.name].x

        return slack_dict
    
    @classmethod
    def relax_constraints(cls, relaxed_model:mip.Model, slack_dict:dict) ->mip.Model:
        """ this method creates a modification of the model `relaxed_model` where all the constraints in the slack_dict are 
        modified in order to add the slack values to make the IIS disappear

        Args:
            relaxed_model (mip.Model): model to relax
            slack_dict (dict): pairs {constraint_name: slack_var.value}

        Returns:
            mip.Model: a modification of the original model where all the constraints are modified with the slack values
        """        
        for crt_name in slack_dict.keys():
            crt_original = relaxed_model.constr_by_name(crt_name)

            relax_expr =  crt_original.expr + slack_dict[crt_name]
            relaxed_model.add_constr(relax_expr, name=crt_original.name)
            relaxed_model.remove(crt_original) # remove constraint 

        return relaxed_model
