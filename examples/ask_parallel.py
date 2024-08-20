import sys

import numpy as np
from cmaes import CMA, SepCMA, XNES, DXNESIC


def ellipsoid(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])

import time

def main():
    otype = 'cma' if len(sys.argv) < 2 else sys.argv[1]
    dim = 1500 if len(sys.argv) < 3 else int(sys.argv[2])
    popsize = 200 if len(sys.argv) < 4 else int(sys.argv[3])
    parallel = len(sys.argv) > 4
    print(f'otype={otype}, dim={dim}, popsize={popsize}, parallel={parallel}')

    optimizer = \
        CMA(mean=3 * np.ones(dim), sigma=2.0, population_size=popsize) if otype == 'cma' else \
        SepCMA(mean=3 * np.ones(dim), sigma=2.0, population_size=popsize) if otype == 'sepcma' else \
        XNES(mean=3 * np.ones(dim), sigma=2.0, population_size=popsize) if otype == 'xnes' else \
        DXNESIC(mean=3 * np.ones(dim), sigma=2.0, population_size=popsize) if otype == 'dxnesic' else \
        None

    iter = 1
    while True:
        print('ask...', end='', flush=True)
        t = time.time()
        if parallel:
            x = optimizer.ask(True)
        else:
            x = [optimizer.ask() for _ in range(optimizer.population_size)]
        print(f'done: {time.time() - t}')
        print('eval...', end='', flush=True)
        t = time.time()
        solutions = [(x[i], ellipsoid(x[i])) for i in range(optimizer.population_size)]
        print(f'done: {time.time() - t}')
        print('tell...', end='', flush=True)
        t = time.time()
        optimizer.tell(solutions)
        print(f'done: {time.time() - t}')
        print(f'value: {min([f for _, f in solutions])}, iter={iter}')
        iter += 1

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
