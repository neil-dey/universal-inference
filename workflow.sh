#!/bin/bash


# Code for Section 5.1
python3 final_code/ramdas_comparison.py 0 0
python3 final_code/ramdas_comparison.py 0 1
python3 final_code/ramdas_comparison.py 1 0
python3 final_code/ramdas_comparison.py 1 1

python3 final_code/ramdas_comparison_closedform.py

# Code for Example 1
python3 final_code/kmeans.py

# Code for Example 2
python3 final_code/svm.py

# Code for Exmaple 3
python3 final_code/lr_condition.py
python3 final_code/lr_samplesize.py 0
python3 final_code/lr_powewr.py

# Code for Example 4
python3 final_code/bestcase.py

# Code for Example 5
python3 final_code/reroll.py

# Code for Example 6
python3 final_code/cherrypicking.py 0
python3 final_code/cherrypicking.py 1

# The data for Section 6.1 are found in millikan.xlsx
python3 final_code/millikan.py

# Code for Section 6.2
python3 final_code/real_quantile.py True
python3 final_code/real_quantile.py False


# Code for Figure S.1 (Figure 12 in the ArXiv version)
for i in {1..99}
do
    python3 final_code/rsafe.py $i
done

# Code for Example S.1 (Example 7 in the ArXiv version)
python3 final_code/heavy_tail.py

# Code for Example S2. (Example 8 in the ArXiv version)
python3 final_code/quantile.py

# Code for Example S.3 (Example 9 in the ArXiv version)
python3 final_code/anova.py
python3 final_code/ci_width.py
