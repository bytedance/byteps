from bytescheduler.common.search import BayesianSearch


def test_bayesian():
    space = {
        "credit": (0, 100),
    }
    steps = 15
    tuner = BayesianSearch(space, steps)
    x = {
        "credit": 0,
    }
    while True:
        y = (x["credit"]-20)**2 + 5
        tuner.put(x, y)
        x, stop = tuner.step()
        if stop:
            print('optimal point is {}'.format(x))
            return

if __name__ == '__main__':
    test_bayesian()