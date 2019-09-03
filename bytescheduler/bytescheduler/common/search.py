from __future__ import absolute_import
from abc import abstractmethod
import collections
import logging
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import random


class Search(object):
    """ Search hyperparameters within space using max_num_steps.
    space is a dict where keys are parameter name and values are two-element tuples, representing the range of each parameter
    e.g., space = {'partition': (2, 4), 'credit': (0, 3)}
    """
    def __init__(self, space, max_num_steps=15, logger=None):
        assert isinstance(space, dict) and len(space) > 0

        # An x:y dict where x is the variable to be tuned and y is typically the time of one iteration.
        # x can be a combination of multiple variables, while y is a scalar.
        self._values = collections.OrderedDict()

        self._space = space
        self._max_num_steps = max_num_steps
        if logger is None:
            self._logger = logging.getLogger("ByteScheduler")
        else:
            self._logger = logger

        # Count the number of search steps.
        self._num_steps = 0
        self._is_first_point = True

        # The optimal hyper-parameters.
        self._opt_point = None
        self._stop = False

        # Do some initialization work
        self._init()

    def _dict_to_tuple(self, source):
        """Transform a dict to tuple"""
        return tuple(source.items())

    def _tuple_to_dict(self, source):
        """Transform a tuple to dict"""
        dest = dict()
        for k, v in source:
            dest[k] = v
        return dest

    def step(self):
        """Run one search step and return next data point (i.e., a dict) for search and whether to stop,
        None means the end of search.

        Returns:
            point: a dict of hyper-parameter configuration
            _stop: a boolean value indicating whether tuning is over or not.
        """

        point = None
        if self._num_steps > self._max_num_steps:
            self._stop = True
            point = self._opt_point
        else:
            self._num_steps += 1
            point = self._step()
        point = dict(point) if point is not None else None
        self._logger.info("Auto-tuning suggests config {} at tuning step {}".format(point, self._num_steps))
        return point, self._stop

    def put(self, var, value):
        """Add the average training time of one step of a configuration"""
        # drop first point as it is typically not accurate.
        if self._is_first_point:
            self._is_first_point = False
        else:
            self._logger.info("Auto-tuning got a point {} with value {}".format(var, value))
            self._values[self._dict_to_tuple(var)] = value
            self._cb(var, value)

    def get(self):
        """Return all tried points"""
        return self._values

    @abstractmethod
    def _step(self):
        """Run one search step and return hyper-parameters"""
        pass

    @abstractmethod
    def _init(self):
        """Some search algorithm may need additional initialization"""
        pass

    @abstractmethod
    def _cb(self, var, value):
        """Some search algorithm may need the value of most recent search step."""
        pass


class BayesianSearch(Search):
    """
    Use Bayesian Optimization to maximize training speed and the decision variables are partition size and credit size.
    """
    def __init__(self, space, max_num_steps):
        super(self.__class__, self).__init__(space, max_num_steps)

        # For UCB acquisition function, smaller kappa prefers exploitation (e.g., 1.0), larger kappa prefers
        # exploration (e.g., 10.0). For EI or PI acquisition function, smaller xi prefers exploitation (e.g., 0.0),
        # larger xi prefers exploration (e.g., 0.1).
        # Check https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation_vs_exploration.ipynb.
        # self._utility = UtilityFunction(kind='ucb', kappa=2.5, xi=0.0)
        self._utility = UtilityFunction(kind='ei', kappa=0.0, xi=0.1)
        self._opt = BayesianOptimization(
            f=None,
            pbounds=self._space,
        )
        self._next_point = None
        self._logger.info("Bayesian Search is enabled, space {}, max_num_steps {}.".format(space, max_num_steps))

    def _init(self):
        """Init random points for probing"""
        self._init_points = []
        for _ in range(10):
            point = {}
            for k, v in self._space.items():
                left, right = v
                point[k] = int(10 ** 6 * (random.random() * (right - left) + left)) / float(10 ** 6)
            self._init_points.append(point)

    def _step(self):
        """Run one step of BO tuning.

        Returns:
            A dict of hyper-parameters suggested by BO as next step configuration.
        """
        if self._num_steps < self._max_num_steps:
            if self._init_points:
                next_point = self._init_points.pop(0)
            else:
                next_point = self._opt.suggest(self._utility)
                for k, v in next_point.items():
                    next_point[k] = int(10**6 * (v)) / float(10**6)
                while self._dict_to_tuple(next_point) in self._values:
                    next_point = self._opt.suggest(self._utility)
            return next_point
        elif self._num_steps == self._max_num_steps:
            self._stop = True
            self._logger.info("Best parameters {}".format(self._opt.max))
            self._opt_point = self._opt.max['params']
            return self._opt_point
        else:
            return

    def _cb(self, var, value):
        """Register a point in BO optimizer

        Arguments:
            var: A dict of hyper-parameters.
            value: The average training time of one step when using the `var` hyper-parameter.
        """
        self._logger.debug('Got point {}: {}'.format(var, value))
        try:
            self._opt.register(
                params=var,
                target=-value,
            )
        except KeyError:
            pass
