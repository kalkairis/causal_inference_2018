# TODO: return a dataframe where last two columns are T and Y
# TODO: A frame of X,effects_T, effects_Y
# TODO: Add a second dataframe of Unobserveds
# TODO: feature types (Z, U, X_M (mediator), X_C (collider), M, W)
# TODO: add true ATE
# TODO: move random seed outside of kang shafer
from src.simulation_functions import kang_shafer_data


class Simulator:
    def __init__(self, sim_type, *args, **kwargs):
        """
        Creates a simulator instance.
        A single simulator instance is based on a simulator type and its arguments (and keywords).
        :param sim_type: A simulator type (available: 'kang_shafer', 'ACIC', 'z_bias')
        :param args: Additional arguments for a single simulation.
        :param kwargs: Additional keyword arguments for a single simulation.
        """
        if sim_type not in self.simulation_types:
            raise IOError(f"Available simulation types are {self.simulation_types}")
        self.sim_type = sim_type
        self.sim_args = args
        self.sim_kwargs = kwargs

    simulation_types = ['kang_shafer', 'ACIC', 'z_bias']

    def simulate_single_case(self):
        if self.sim_type == 'kang_shafer':
            tmp = kang_shafer_data(*self.sim_args, **self.sim_kwargs)
            return tmp


if __name__ == "__main__":
    S = Simulator('kang_shafer', num_samples=100)
    tmp1 = S.simulate_single_case()
    tmp2 = S.simulate_single_case()
    print('this was fun and all')
