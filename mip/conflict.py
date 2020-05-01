import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from mip.constants import (
    INF,
    MINIMIZE,
    MAXIMIZE,
    GUROBI,
    CBC,
    CONTINUOUS,
    LP_Method,
    OptimizationStatus,
    SearchEmphasis,
    VERSION,
    BINARY,
    INTEGER,
    CutType,
)
from mip.log import ProgressLog
from mip.model import Model
from mip.lists import ConstrList

from mip.exceptions import (
    InvalidLinExpr,
    InfeasibleSolution,
    SolutionNotAvailable,
)

class ConflictFinder:

    def __init__():
        # TODO
        pass

    def find_iis(self,
                 model: "Model", 
                 method: str = "deletion-filter",) -> ConstrList:
        # check if infeasible 
        assert model.status == InfeasibleSolution, "model is not infeasible"

        if method == "deletion-filter":
            return self.deletion_filter(model)
    
    def deletion_filter(self, 
                        model:"Model")-> ConstrList:
        # get all constraints

        # 1. create a model with all constraints but one 
        # 2. test feasibility, if feasible, return dropped constraint to the set 
        # 2.1 else removed it permanently 

        pass