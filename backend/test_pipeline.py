from app import run_pipeline
result = run_pipeline()
m = result['metrics_summary']
print("=== DRISHTINAV RESULTS ===")
print("Distance:", round(m['total_distance'], 1), "m")
print("DR:   RMSE=", round(m['dr_rmse'],1), "m  Drift=", round(m['dr_drift_pct'],1), "%  Pass=", m['dr_pass'])
print("EKF:  RMSE=", round(m['ekf_rmse'],1), "m  Drift=", round(m['ekf_drift_pct'],1), "%  Pass=", m['ekf_pass'])
print("MAP:  RMSE=", round(m['map_rmse'],1), "m  Drift=", round(m['map_drift_pct'],1), "%  Pass=", m['map_pass'])
print("Speed RMSE:", round(m['speed_rmse'],2), "m/s")
print("Tunnel:", result['tunnel']['start_sec'], "-", result['tunnel']['end_sec'], "sec")
print("GT pts:", len(result['ground_truth']['x']))
