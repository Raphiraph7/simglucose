from simglucose.simulation.scenario import Action, Scenario
import numpy as np
from scipy.stats import truncnorm
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RandomScenario(Scenario):
    def __init__(self, start_time, seed=None):
        Scenario.__init__(self, start_time=start_time)
        self.seed = seed

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)

        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            return Action(meal=self.scenario['meal']['amount'][idx])
        else:
            return Action(meal=0)

    def create_scenario(self):

        time_intervall = 3 # minutes

        scenario = {'meal': {'time': [], 'amount': []}}

        # Meal information given in paper
        time_mu = np.array([7, 14, 21]) * 60
        time_sd = np.array([30,  30, 30])
        amount_mu = [70, 110, 90]
        amount_sd = [7, 11, 9]
        time_meal_intake = 15

        for tm, ts, mm, ms in zip(time_mu, time_sd,
                                        amount_mu, amount_sd):

            time_meal = np.round(self.random_gen.normal(tm, ts))
            amount_meal = max(self.random_gen.normal(mm, ms), 0)
            td = 0
            while td < time_meal_intake:
                scenario['meal']['time'].append(time_meal + td)
                scenario['meal']['amount'].append(amount_meal/(time_meal_intake/time_intervall))
                td += time_intervall
            # scenario['meal']['time'].append(time_meal)
            # scenario['meal']['amount'].append()

        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


if __name__ == '__main__':
    from datetime import time
    from datetime import timedelta
    import copy
    now = datetime.now()
    t0 = datetime.combine(now.date(), time(6, 0, 0, 0))
    t = copy.deepcopy(t0)
    sim_time = timedelta(days=2)

    scenario = RandomScenario(seed=1)
    m = []
    T = []
    while t < t0 + sim_time:
        action = scenario.get_action(t)
        m.append(action.meal)
        T.append(t)
        t += timedelta(minutes=1)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.plot(T, m)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    plt.show()
