"""Single-server scheduling replay of the Azure LLM inference trace."""
import heapq
import sys

import numpy as np
import pandas as pd

PREFILL_COST = 0.02  # one prompt token costs 1/50 of a decode token
QUANTUM = None       # set per run (preemptive policies re-decide at this granularity)


def load(path, rho):
    d = pd.read_csv(path)
    t = pd.to_datetime(d["TIMESTAMP"])
    arr = (t - t.min()).dt.total_seconds().to_numpy()
    order = np.argsort(arr, kind="stable")
    d = d.iloc[order].reset_index(drop=True)
    arr = arr[order]
    work = PREFILL_COST * d["ContextTokens"].to_numpy() + d["GeneratedTokens"].to_numpy()
    span = arr[-1] - arr[0]
    # scale service so that total work / span = rho
    svc = work * (rho * span / work.sum())
    return arr, svc, d["ContextTokens"].to_numpy(), d["GeneratedTokens"].to_numpy()


def nonpreemptive(arr, svc, key):
    """Serve jobs to completion; when free, pick waiting job with smallest key."""
    n = len(arr)
    done = np.empty(n)
    t, i, heap = 0.0, 0, []
    while i < n or heap:
        if not heap:
            t = max(t, arr[i])
        while i < n and arr[i] <= t:
            heapq.heappush(heap, (key[i], i))
            i += 1
        _, j = heapq.heappop(heap)
        t += svc[j]
        done[j] = t
    return done - arr


def preemptive(arr, svc, keyfn, quantum):
    """Re-decide at every arrival and every `quantum` of service.
    keyfn(j, attained, remaining) -> priority (smaller first)."""
    n = len(arr)
    rem = svc.copy()
    att = np.zeros(n)
    done = np.empty(n)
    active = []  # list of job ids
    t, i = 0.0, 0
    while i < n or active:
        if not active:
            t = max(t, arr[i])
        while i < n and arr[i] <= t:
            active.append(i)
            i += 1
        j = min(active, key=lambda k: keyfn(k, att[k], rem[k]))
        next_arr = arr[i] if i < n else np.inf
        run = min(rem[j], quantum, max(next_arr - t, 1e-12))
        t += run
        rem[j] -= run
        att[j] += run
        if rem[j] <= 1e-12:
            done[j] = t
            active.remove(j)
    return done - arr


def main(path, rho):
    arr, svc, ctx, gen = load(path, rho)
    q = float(np.median(svc)) / 4
    res = {
        "FCFS": nonpreemptive(arr, svc, np.arange(len(arr))),
        "Prompt-length SJF (heuristic)": nonpreemptive(arr, svc, ctx),
        "LAS / tokens-generated-so-far (no prediction)": preemptive(arr, svc, lambda k, a, r: a, q),
        "Oracle SJF (knows output length)": nonpreemptive(arr, svc, svc),
        "Oracle SRPT (knows output length)": preemptive(arr, svc, lambda k, a, r: r, q),
    }
    fcfs = res["FCFS"].mean()
    orc = res["Oracle SRPT (knows output length)"].mean()
    print(f"\n{path}  rho={rho}  n={len(arr)}")
    print(f"{'policy':48s} {'mean':>9s} {'p99':>9s} {'gain captured':>14s}")
    for k, v in res.items():
        cap = (fcfs - v.mean()) / (fcfs - orc) * 100 if fcfs != orc else float("nan")
        print(f"{k:48s} {v.mean():9.2f} {np.percentile(v, 99):9.2f} {cap:13.1f}%")
    return res


def plot(path, rhos, results):
    import matplotlib.pyplot as plt

    policies = list(results[rhos[0]].keys())
    fig, (ax_mean, ax_p99) = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.8 / len(rhos)
    x = np.arange(len(policies))
    for i, rho in enumerate(rhos):
        means = [results[rho][p].mean() for p in policies]
        p99s = [np.percentile(results[rho][p], 99) for p in policies]
        offset = (i - (len(rhos) - 1) / 2) * width
        ax_mean.bar(x + offset, means, width, label=f"rho={rho}")
        ax_p99.bar(x + offset, p99s, width, label=f"rho={rho}")

    for ax, title in ((ax_mean, "mean latency"), (ax_p99, "p99 latency")):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("seconds")
        ax.legend()

    fig.suptitle(path)
    fig.tight_layout()
    fig.savefig("results.png", dpi=150)


if __name__ == "__main__":
    path = sys.argv[1]
    args = sys.argv[2:]
    want_plot = "--plot" in args
    if want_plot:
        args.remove("--plot")
    rhos = [float(x) for x in args] or [0.7, 0.9]
    all_res = {rho: main(path, rho) for rho in rhos}
    if want_plot:
        plot(path, rhos, all_res)
