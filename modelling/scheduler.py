import pandas as pd
import numpy as np
import pyomo.environ as pe
import pyomo.gdp as pyogdp
from pyomo.core.base.set_types import Any
import os


# check with Taha if code is too similar to Alstom?
class TheatreScheduler:

    def __init__(self, case_file_path, session_file_path):
        """
        Read case and session data into Pandas DataFrames
        Args:
            case_file_path (str): path to case data in CSV format
            session_file_path (str): path to theatre session data in CSV format
        """
        try:
            self.df_cases = pd.read_csv(case_file_path)
        except FileNotFoundError:
            print("Case data not found.")

        try:
            self.df_sessions = pd.read_csv(session_file_path)
        except FileNotFoundError:
            print("Session data not found")

    def _generate_case_durations(self):
        """
        Generate mapping of cases IDs to median case time for the procedure
        Returns:
            (dict): dictionary with CaseID as key and median case time (mins) for procedure as value
        """
        return pd.Series(self.df_cases["Median Duration"].values, index=self.df_cases["CaseID"]).to_dict()

    def _generate_session_durations(self):
        """
        Generate mapping of all theatre sessions IDs to session duration in minutes
        Returns:
            (dict): dictionary with SessionID as key and session duration as value
        """
        return pd.Series(self.df_sessions["Duration"].values, index=self.df_sessions["SessionID"]).to_dict()

    def create_model(self):
        model = pe.ConcreteModel()

        # Model Data
        model.CASES = pe.Set(initialize=self.df_cases["CaseID"].tolist())
        model.SESSIONS = pe.Set(initialize=self.df_sessions["SessionID"].tolist())
        model.TASKS = pe.Set(initialize=model.CASES * model.SESSIONS, dimen=2)
        model.CASE_DURATIONS = pe.Param(model.CASES, initialize=self._generate_case_durations())
        model.SESSION_DURATIONS = pe.param(model.SESSIONS, initalize=self._generate_session_durations())
        model.M = pe.Param(initialize=1e7)  # big M
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


if __name__ == "__main__":
    case_path = os.path.join(os.path.dirname(os.getcwd()), "data", "case_data.csv")
    session_path = os.path.join(os.path.dirname(os.getcwd()), "data", "session_data.csv")
    scheduler = TheatreScheduler(case_file_path=case_path, session_file_path=session_path)
