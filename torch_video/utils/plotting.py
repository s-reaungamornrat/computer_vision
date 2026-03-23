import warnings
from pathlib import Path

import numpy as np

def plot_results(file:str, dir:str=None):
    """Plot training results from a result CSV files. The function plots by overlay training/val metrics from multiple runs 
    (multiple CSV files) onto a single set of plots for comparison. The plots are stored as 'results.png' in the directory where 
    the CSV is located
    Args:
        file (str, optional): Path to the CSV file containing the training results
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided
    """
    assert any(x is not None for x in [file, dir]),"Must provide either path of CSV via 'file' or directory of CSV via 'dir'"
    
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl
    from scipy.ndimage import gaussian_filter1d
    
    save_dir=Path(file).parent if file else Path(dir)
    files=list(save_dir.glob("result*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot"
    
    loss_keys, metric_keys=[],[]
    for i, f in enumerate(files):
        try:
            data=pl.read_csv(f, infer_schema_length=None)
            if i==0:
                for c in data.columns:
                    if "loss" in c: loss_keys.append(c) # e.g., 'train/loss', 'val/loss'
                    elif "metric" in c: metric_keys.append(c)
                loss_mid, metric_mid=len(loss_keys)//2, len(metric_keys)//2 
                # separate train and val so 'train/loss', 'train/metric1', 'train/metric2',..., 'val/loss', 'val/metric1', 'val/metric2',...
                columns=(loss_keys[:loss_mid]+metric_keys[:metric_mid]+loss_keys[loss_mid:]+metric_keys[metric_mid:])
                # declare fig and ax only for i=0 since we want to overlay training/val metrics from multiple runs (multiple CSV files) 
                # onto a single set of plots for comparison.
                fig, ax = plt.subplots(2, len(columns) // 2, figsize=(len(columns) + 2, 6), tight_layout=True)
                ax = ax.ravel() # convert 2 dim array to 1 dim array
            x = data.select(data.columns[0]).to_numpy().flatten()
            for j, k in enumerate(columns):
                y = data.select(k).to_numpy().flatten().astype("float")
                ax[j].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[j].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[j].set_title(k, fontsize=12)
        except Exception as e:
            warnings.error(f"Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()


def plot_all(file:str, dir:str=None):
    """Plot training results from a result CSV files. The function plots by overlay training/val metrics from multiple runs 
    (multiple CSV files) onto a single set of plots for comparison. The plots are stored as 'results.png' in the directory where 
    the CSV is located
    Args:
        file (str, optional): Path to the CSV file containing the training results
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided
    """
    assert any(x is not None for x in [file, dir]),"Must provide either path of CSV via 'file' or directory of CSV via 'dir'"
    
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl
    from scipy.ndimage import gaussian_filter1d
    
    save_dir=Path(file).parent if file else Path(dir)
    files=list(save_dir.glob("result*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot"
    
    loss_keys, metric_keys=[],[]
    for i, f in enumerate(files):
        try:
            data=pl.read_csv(f, infer_schema_length=None)
            if i==0:
                columns=list(data.columns)
                if 'epoch' in columns: columns.remove('epoch')
                half_n_columns=int(np.ceil(len(columns)/2))
                #print(f"{columns=}, {len(columns)=}, {half_n_columns =}")
                fig, ax = plt.subplots(2, half_n_columns, figsize=(2*half_n_columns + 2, 6), tight_layout=True)
                ax = ax.ravel() # convert 2 dim array to 1 dim array
            x = data.select(data.columns[0]).to_numpy().flatten()
            for j, k in enumerate(columns):
                #print(f"{j=}, {k=}")
                y = data.select(k).to_numpy().flatten().astype("float")
                ax[j].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[j].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[j].set_title(k, fontsize=12)
                if j==len(columns)-1: break
        except Exception as e:
            raise RuntimeError(f"Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()