
"""Tests for Python-MIP conflict-finder"""
import pytest
from mip import  CBC, GUROBI
from os import environ

TOL = 1e-4

SOLVERS = [CBC]
if "GUROBI_HOME" in environ:
    SOLVERS += [GUROBI]


@pytest.mark.parametrize("solver", SOLVERS)
def test_conflict_refiner(solver: str):
    # TODO
    pass

# test feasible model 
# test infeasible simple model 
# test a mip
# test a single IIS model in all methods 

