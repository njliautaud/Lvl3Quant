import sys
sys.path.insert(0, '/home/nick/Lvl3Quant/scripts')
import train_streaming_continuation_v1 as mod
mod.HORIZONS = [30]
mod.EXPERIMENT_NAME = 'streaming_continuation_v1'
print('Overriding HORIZONS to [30]')
mod.main()
