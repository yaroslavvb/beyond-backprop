import cProfile
import pstats
import sys
import os

# Add parent directory to path to import attention_solve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_solve import train, test_initial_lr_stability

print("================ PROFILING STABILITY TEST ================")
profiler = cProfile.Profile()
profiler.enable()
test_initial_lr_stability()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)

print("\n================ PROFILING MAIN TRAINING ================")
profiler = cProfile.Profile()
profiler.enable()
train()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(35)
