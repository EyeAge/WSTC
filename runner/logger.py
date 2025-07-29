# logger.py  (单独文件，所有脚本 import)
import os, time, uuid, pandas as pd, json, numpy as np

def log_run(result_row: dict,
            y_true: np.ndarray | None = None,
            y_pred: np.ndarray | None = None,
            cfg:   dict | None = None,
            root:  str = "experiments"):
    """把一次实验的信息写进 CSV，并可选保存预测与配置。"""
    os.makedirs(root,            exist_ok=True)
    os.makedirs(f"{root}/preds", exist_ok=True)
    os.makedirs(f"{root}/cfgs",  exist_ok=True)

    # csv_path = f"{root}/runs.csv"
    csv_path = os.path.join(root, result_row["dataset"], result_row["model"], "runs.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame([result_row]).to_csv(
        csv_path, mode="a", index=False,
        header=not os.path.exists(csv_path))

    run_id = result_row["run_id"]
    if y_true is not None and y_pred is not None:
        np.savez_compressed(f"{root}/preds/{run_id}.npz",
                            y_true=y_true, y_pred=y_pred)
    if cfg is not None:
        with open(f"{root}/cfgs/{run_id}.json", "w") as f:
            json.dump(cfg, f, indent=2)
