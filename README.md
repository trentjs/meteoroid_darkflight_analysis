# Meteoroid Darkflight Analysis

The meteoroid darkflight algorithm developed for the DFN as part of my PhD.

For general instruction / help, run:
```
python3 DarkFlightTrent.py -h
```

For a single mean darkflight curtin, run:
```
python3 DarkFlightTrent.py -e data/DN161031_01_triangulation_all_timesteps.ecsv -w data/profile_DN161031_1200_model_start_2016-10-31_1200.csv -v raw -m 1.796 -s 1.78 -g 315
```

For the monte-carlo darkflight distribution, run:
```
mpirun -n 4 python3 DarkFlightTrent.py -e data/DN161031_01_triangulation_all_timesteps.ecsv -w data/profile_DN161031_1200_model_start_2016-10-31_1200.csv -v raw -m 1.796 -s 1.78 -g 315 -mc 500
```
