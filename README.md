# Automated ML Training Infrastructure


This project simulates an ML training workflow and adds basic observability/monitoring around it.
It trains a simple binary classifier and logs epoch metrics so that the training behavior can be viewed after the run. Probably going to add a KPI indicator or some type of visualizer in future renditions.
Each epoch is logged to a .jsonl file with timestamp, epoch number, loss, accuracy, learning rate, and runtime.
The monitor also checks for alert conditions such as NaN loss, repeatedly increasing loss (possible divergence), and low accuracy after multiple epochs.
Finally, a summary report is generated with best loss, best accuracy, total runtime, and any triggered alerts.
This project is meant to demonstrate Python automation, monitoring, observability, and reporting for ML workflows.

Run with: python train.py
Outputs are written to the logs/directory (epoch log + final summary JSON).