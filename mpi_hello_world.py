#!/usr/bin/env python

from mpi4py import MPI
import sys

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    sys.stdout.write(
        "Hello from process %d of %d on %s.\n" % (rank, size, name))

if __name__ == '__main__':
    main()
