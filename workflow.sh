#!/bin/bash

# Code for Section 5
python3 final_code/ramdas_comparison.py 0 0
python3 final_code/ramdas_comparison.py 0 1
python3 final_code/ramdas_comparison.py 1 0
python3 final_code/ramdas_comparison.py 1 1

python3 final_code/ramdas_comparison_closedform.py

# Code for Example 1
python3 final_code/bestcase_offline.py
python3 final_code/bestcase_online.py

# Code for Example 2
python3 final_code/reroll_online.py
python3 final_code/reroll_offline.py

# Code for Example 3
python3 final_code/cherrypicking_online.py 0
python3 final_code/cherrypicking_online.py 1
python3 final_code/cherrypicking_offline.py 0
python3 final_code/cherrypicking_offline.py 1

# Code for Section 6.2
python3 final_code/kmeans.py

# The calculations for Section 7 are found in millikan.xlsx
