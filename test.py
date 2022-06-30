from python_block_matching.utils import *
from python_block_matching.cost_functions import *
from python_block_matching.algorithms import *

l = lambda *args: two_dimensional_logarithmic_search(*args, cost_function='MSE')
test_algorithm('allmost360.h264', l)
