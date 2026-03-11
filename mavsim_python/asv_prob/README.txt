ASV EKF Starter Code (Problem 3)

Files
-----
- asv_ekf_run.py
    Loads the dataset, runs your EKF, prints metrics, and makes the required plots.
- asv_observer.py
    EKF skeleton. Fill in the TODO sections:
    * parameters Ts, aV, bV, kchi
    * noise covariances Q and R
    * initial covariance P
    * Jacobians A and C
    * propagation and correction steps

Data
----
Place the provided data file in this same folder as:
    asv_data.txt

Run
---
    python asv_ekf_run.py
