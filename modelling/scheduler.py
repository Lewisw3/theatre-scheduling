import pandas as pd
import numpy as np
import pyomo.environ as pe
import pyomo.gdp as pyogdp
from pyomo.core.base.set_types import Any


# check with Taha if code is too similar to Alstom?
class TheatreScheduler:

    def __init__(self):
        pass

    def _generate_cases(self):
        pass

    def create_model(self):
        model = pe.ConcreteModel()

        # Model Data
        model.CASES = pe.Set()  # indexed by (caseID, sessionID)
        model.CASE_DURATIONS = pe.Param()  # median case times
        model.SESSION_DURATIONS = pe.param()  # session durations
        model.M = pe.Param()  # big M
        max_util = 0.85

        # Decision Variables
        model.SESSION_ASSIGNED = pe.Var()
        model.CASE_START_TIME = pe.Var()
        model.UTILISATION = pe.Var()

        # Objective
        def objective_function(model):
            pass
        model.OBJECTIVE = pe.Objective()

        # Constraints
        # Case start time must be after start time of assigned theatre session
        def case_start_time():
            pass
        model.CASE_START = pe.Constraint()

        # Case end time must be before end time of assigned theatre session
        def case_end_time():
            pass
        model.CASE_END_TIME = pe.Constraint()

        # Cases can be assigned to a maximum of one session
        def session_assignment():
            pass
        model.SESSION_ASSIGNMENT = pe.Constraint()

        def disjunctions():
            pass
        model.DISJUNCTIONS = pyogdp.Disjunction()

        def theatre_util():
            pass
        model.THEATRE_UTIL = pe.Constraint()

        return model

    def solve(self):
        pass
