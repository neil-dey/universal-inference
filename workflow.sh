#!/bin/bash


# Code for Section 5.1
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


# Code for Example 4
for i in {80..99}
do
    python3 final_code/anova.py $i
done

# Code for Example S.1 (Example 5 in the ArXiv version)
python3 final_code/heavy_tail.py

# Code for Example S2. (Example 6 in the ArXiv version)
python3 final_code/restricted_mean.py on
python3 final_code/restricted_mean.py off

# Code for Example 5 (Example 7 in ArXiv verison)
python3 final_code/kmeans.py

# The calculations for Section 6.1 are found in millikan.xlsx

# Code for Section 6.2
python3 final_code/real_quantile.py True
python3 final_code/real_quantile.py False

