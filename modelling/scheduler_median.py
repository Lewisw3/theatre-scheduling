import pandas as pd
import numpy as np
import pyomo.environ as pe
import pyomo.gdp as pyogdp
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product


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

        self.model = self.create_model()

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

    def _generate_session_start_times(self):
        """
        Generate mapping from SessionID to session start time
        Returns:
            (dict): dictionary with SessionID as key and start time in minutes since midnight as value
        """
        # Convert session start time from HH:MM:SS format into seconds elapsed since midnight
        self.df_sessions.loc[:, "Start"] = pd.to_timedelta(self.df_sessions["Start"])
        self.df_sessions.loc[:, "Start"] = self.df_sessions["Start"].dt.total_seconds() / 60
        return pd.Series(self.df_sessions["Start"].values, index=self.df_sessions["SessionID"]).to_dict()

    def _generate_disjunctions(self):
        """
        #TODO
        Returns:
            disjunctions (list): list of tuples containing disjunctions
        """
        cases = self.df_cases["CaseID"].to_list()
        sessions = self.df_sessions["SessionID"].to_list()
        disjunctions = []
        for (case1, case2, session) in product(cases, cases, sessions):
            if (case1 != case2) and (case2, case1, session) not in disjunctions:
                disjunctions.append((case1, case2, session))

        return disjunctions

    def create_model(self):
        model = pe.ConcreteModel()

        # Model Data
        model.CASES = pe.Set(initialize=self.df_cases["CaseID"].tolist())
        model.SESSIONS = pe.Set(initialize=self.df_sessions["SessionID"].tolist())
        model.TASKS = pe.Set(initialize=model.CASES * model.SESSIONS, dimen=2)
        model.CASE_DURATION = pe.Param(model.CASES, initialize=self._generate_case_durations())
        model.SESSION_DURATION = pe.Param(model.SESSIONS, initialize=self._generate_session_durations())
        model.SESSION_START_TIME = pe.Param(model.SESSIONS, initialize=self._generate_session_start_times())
        model.DISJUNCTIONS = pe.Set(initialize=self._generate_disjunctions(), dimen=3)

        ub = 1440  # seconds in a day
        model.M = pe.Param(initialize=1e3*ub)  # big M
        max_util = 0.85
        num_sessions = self.df_sessions.shape[0]

        # Decision Variables
        model.SESSION_ASSIGNED = pe.Var(model.TASKS, domain=pe.Binary)
        model.CASE_START_TIME = pe.Var(model.TASKS, bounds=(0, ub), within=pe.PositiveReals)
        model.UTILISATION = pe.Var(model.SESSIONS, bounds=(0, 1), within=pe.PositiveReals)
        model.MEDIAN_UTIL = pe.Var(bounds=(0, ub), within=pe.PositiveReals)
        model.DUMMY_BINARY = pe.Var(model.SESSIONS, domain=pe.Binary)
        model.CANCEL_SESSION = pe.Var(model.SESSIONS, domain=pe.Binary, within=pe.PositiveReals)

        # Objective
        def objective_function(model):
            #return pe.summation(model.UTILISATION)
            return model.MEDIAN_UTIL
        model.OBJECTIVE = pe.Objective(rule=objective_function, sense=pe.maximize)

        # Constraints

        # TODO add constraint to complete before deadline if it is assigned
        # TODO add constraint to make tasks follow each other without gaps?

        # Case start time must be after start time of assigned theatre session
        def case_start_time(model, case, session):
            return model.CASE_START_TIME[case, session] >= model.SESSION_START_TIME[session] - \
                   ((1 - model.SESSION_ASSIGNED[(case, session)])*model.M)
        model.CASE_START = pe.Constraint(model.TASKS, rule=case_start_time)

        # Case end time must be before end time of assigned theatre session
        def case_end_time(model, case, session):
            return model.CASE_START_TIME[case, session] + model.CASE_DURATION[case] <= model.SESSION_START_TIME[session] + \
                   model.SESSION_DURATION[session]*max_util + ((1 - model.SESSION_ASSIGNED[(case, session)]) * model.M)
        model.CASE_END_TIME = pe.Constraint(model.TASKS, rule=case_end_time)

        # Cases can be assigned to a maximum of one session
        def session_assignment(model, case):
            return sum([model.SESSION_ASSIGNED[(case, session)] for session in model.SESSIONS]) <= 1
        model.SESSION_ASSIGNMENT = pe.Constraint(model.CASES, rule=session_assignment)

        def no_case_overlap(model, case1, case2, session):
            return [model.CASE_START_TIME[case1, session] + model.CASE_DURATION[case1] <= model.CASE_START_TIME[case2, session] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M),
                    model.CASE_START_TIME[case2, session] + model.CASE_DURATION[case2] <= model.CASE_START_TIME[case1, session] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M)]

        model.DISJUNCTIONS_RULE = pyogdp.Disjunction(model.DISJUNCTIONS, rule=no_case_overlap)

        def theatre_util(model, session):
            return model.UTILISATION[session] == (1 / model.SESSION_DURATION[session]) * \
                   sum([model.SESSION_ASSIGNED[case, session]*model.CASE_DURATION[case] for case in model.CASES])
        model.THEATRE_UTIL = pe.Constraint(model.SESSIONS, rule=theatre_util)

        def cancel_sessions(model, session): # TODO
            return model.CANCEL_SESSION[session] <= 1 - model.M*sum([model.SESSION_ASSIGNED[case, session] for case in model.CASES])
        model.SET_CANCEL_SESSIONS = pe.Constraint(model.SESSIONS, rule=cancel_sessions)

        def force_cancel_sessions(model, session):
            return sum([model.SESSION_ASSIGNED[case, session] for case in model.CASES]) <= 0 + model.M*(1-model.CANCEL_SESSION[session])
        #model.FORCE_CANCEL_SESSIONS = pe.Constraint(model.SESSIONS, rule=force_cancel_sessions)

        def set_dummy_variable(model):
            return sum([model.DUMMY_BINARY[session] for session in model.SESSIONS]) == np.floor(num_sessions/2)
        model.FLOOR = pe.Constraint(rule=set_dummy_variable)

        def set_median_util(model, session):
            return model.MEDIAN_UTIL <= model.UTILISATION[session] + model.DUMMY_BINARY[session]*model.M
        model.SET_MEDIAN_UTIL = pe.Constraint(model.SESSIONS, rule=set_median_util)

        pe.TransformationFactory("gdp.bigm").apply_to(model)

        return model

    def solve(self, solver_name, options=None, solver_path=None):

        if solver_path is not None:
            solver = pe.SolverFactory(solver_name, executable=solver_path)
        else:
            solver = pe.SolverFactory(solver_name)

        # TODO remove - too similar to alstom
        if options is not None:
            for key, value in options.items():
                solver.options[key] = value

        solver_results = solver.solve(self.model, tee=True)

        results = [{"Case": case,
                    "Session": session,
                    "Start": self.model.CASE_START_TIME[case, session](),
                    "Assignment": self.model.SESSION_ASSIGNED[case, session]()}
                   for (case, session) in self.model.TASKS]

        self.df_times = pd.DataFrame(results)

        all_cases = self.model.CASES.value_list
        cases_assigned = []
        cases_missed = []
        for (case, session) in self.model.SESSION_ASSIGNED:
            if self.model.SESSION_ASSIGNED[case, session] == 1:
                cases_assigned.append(case)

        cases_missed = list(set(all_cases).difference(cases_assigned))
        print("Number of cases assigned = {} out of {}:".format(len(cases_assigned), len(all_cases)))
        print("Cases assigned: ", cases_assigned)
        print("Number of cases missed = {} out of {}:".format(len(cases_missed), len(all_cases)))
        print("Cases missed: ", cases_missed)
        self.model.UTILISATION.pprint()
        print("Total Utilisation = {}".format(sum(self.model.UTILISATION.get_values().values())))
        print("Number of constraints = {}".format(solver_results["Problem"].__getitem__(0)["Number of constraints"]))
        #self.model.SESSION_ASSIGNED.pprint()
        #print(self.df_times.to_string())
        self.draw_gantt()

    def draw_gantt(self):

        df = self.df_times[self.df_times["Assignment"] == 1]
        cases = sorted(list(df['Case'].unique()))
        sessions = sorted(list(df['Session'].unique()))

        bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
        text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center'}
        colors = cm.Dark2.colors

        df.sort_values(by=['Case', 'Session'])
        df.set_index(['Case', 'Session'], inplace=True)

        fig, ax = plt.subplots(1, 1)
        for c_ix, c in enumerate(cases, 1):
            for s_ix, s in enumerate(sessions, 1):
                if (c, s) in df.index:
                    xs = df.loc[(c, s), 'Start']
                    xf = df.loc[(c, s), 'Start'] + \
                         self.df_cases[self.df_cases["CaseID"] == c]["Median Duration"]
                    ax.plot([xs, xf], [s] * 2, c=colors[c_ix % 7], **bar_style)
                    ax.text((xs + xf) / 2, s, c, **text_style)

        ax.set_title('Session Schedule')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sessions')
        ax.grid(True)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    case_path = os.path.join(os.path.dirname(os.getcwd()), "data", "case_data_long.csv")
    session_path = os.path.join(os.path.dirname(os.getcwd()), "data", "session_data.csv")
    cbc_path = "C:\\Users\\LONLW15\\Documents\\Linear Programming\\Solvers\\cbc.exe"

    options = {"seconds": 30}
    scheduler = TheatreScheduler(case_file_path=case_path, session_file_path=session_path)
    scheduler.solve(solver_name="cbc", solver_path=cbc_path, options=options)
