import pprint

import tomli

from .plot import plot_accuracy_curve_by_exp_group


class ExperimentCfg:
    def __init__(self, cfg: dict, name: str = None, exp_value: str = None):
        """
        Initialize the experiment configuration.

        Args:
            cfg (dict): configuration dictionary
            name (str): name of the experiment
            exp_value (str): value of the experiment
        """
        self.trainer = cfg.copy()
        self.name = name
        self.exp_value = exp_value
        self.reorganize()

    def reorganize(self):
        """
        Reorganize the configuration dictionary.
        """
        self.epochs = self.trainer.pop('epochs')
        self.batch_size = self.trainer.pop('batch_size')
        self.use_aug = self.trainer.pop('use_aug')

    def __repr__(self) -> str:
        """
        Return the string representation of the configuration.

        Returns:
            str: string representation of the configuration
        """
        return str(self.trainer)


class ExperimentGroup:
    def __init__(self, name: str, default_params: dict, exp_params: dict):
        """
        Initialize the experiment group.

        Args:
            name (str): name of the experiment group
            default_params (dict): default parameters
            exp_params (dict): experiment parameters
        """
        self.name = name
        self.params = default_params
        self.exp_column = exp_params['experiment']
        self.exp_value = exp_params['value']
        self._exp_results = {}
        self._gen = self.__generate()

    def __iter__(self) -> 'ExperimentGroup':
        """
        Return the iterator of the experiment group.

        Returns:
            ExperimentGroup: iterator of the experiment group
        """
        return self

    def __next__(self) -> ExperimentCfg:
        """
        Return the next experiment configuration.

        Returns:
            ExperimentCfg: next experiment configuration
        """
        return next(self._gen)

    def __generate(self) -> ExperimentCfg:
        """
        Generate the experiment configuration.

        Returns:
            ExperimentCfg: experiment configuration

        Yields:
            Iterator[ExperimentCfg]: iterator of the experiment configuration
        """
        for exp_param in self.exp_value:
            cfg = self.params
            cfg[self.exp_column] = exp_param
            yield ExperimentCfg(cfg, name=self.name, exp_value=exp_param)
        else:
            plot_accuracy_curve_by_exp_group(fname=f'result/{self.name}.png', title=self.name, **self._exp_results)
            return StopIteration

    @property
    def results(self) -> dict:
        """
        Return the results of the experiment group.

        Returns:
            dict: results of the experiment group
        """
        return self._exp_results

    @results.setter
    def results(self, result: dict):
        """
        Set the results of the experiment group.

        Args:
            result (dict): results of the experiment group
        """
        for key, value in result.items():
            self._exp_results[f"{self.params[self.exp_column]}_{key}"] = value


class ExperimentManager:
    def __init__(self, config_filename: str):
        """
        Initialize the experiment manager.

        Args:
            config_filename (str): configuration filename
        """
        self.config_filename = config_filename
        self.load_and_parse_config()
        self.summary_result = dict()

    def __call__(self, function):
        """
        Decorate the function.

        Args:
            function (function): function to be decorated

        """
        def wrapper():
            for experiment in self.experiments:
                if experiment['enable'] == True:
                    self.set_experiment_group(experiment)
                    for exp in self.experiment_group:
                        pprint.pprint(exp)
                        self.experiment_group.results = function(exp)
                    self.summary_result[experiment['name']] = dict()
                    for key, value in self.experiment_group.results.items():
                        self.summary_result[experiment['name']][key] = value[-1]
            pprint.pprint(self.summary_result)
        return wrapper

    def load_and_parse_config(self):
        """
        Load and parse the configuration.
        """
        self.load_config()
        self.parse_config()

    def load_config(self):
        """
        Load the configuration.
        """
        with open(self.config_filename, mode="rb") as fp:
            self.config = tomli.load(fp)

    def parse_config(self):
        """
        Parse the configuration.
        """
        self.experiments = self.config['experiment']
        self.defaults = self.config['default']
        self.specifics = self.config['specific']

    def set_experiment_group(self, experiment: dict):
        """
        Set the experiment group.

        Args:
            experiment (dict): experiment group
        """
        name = experiment['name']
        model_name = experiment['model_name']
        default_params = self.defaults | self.specifics[model_name]
        for key in experiment.keys():
            if key in default_params:
                default_params[key] = experiment[key]
        exp_params = {
            'experiment': experiment['experiment'],
            'value': experiment['value']
        }
        self.experiment_group = ExperimentGroup(name, default_params, exp_params)
