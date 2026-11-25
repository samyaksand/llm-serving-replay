# LLM serving replay

Does knowing a request's prompt length help schedule LLM inference requests, compared with knowing its output length?

## Finding

At 90% load, an oracle that knows output length roughly halves mean latency compared with first-come-first-served. Ordering by prompt length instead captures none of that gain and ends up worse than first-come-first-served. Even the oracle has a worse p99 than first-come-first-served. Prompt length and output length are nearly uncorrelated in this trace (Spearman 0.07), which is why prompt length doesn't work as a stand-in for output length.

## Figure

![results](results.png)

## Run it

Download the trace CSV from the Azure Public Dataset:

https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_conv.csv

Install dependencies, then run:

```
pip install -r requirements.txt
python llm_sched_sim.py AzureLLMInferenceTrace_conv.csv 0.7 0.9 --plot
```

## Results

```
AzureLLMInferenceTrace_conv.csv  rho=0.7  n=19366
policy                                                mean       p99  gain captured
FCFS                                                  0.40      2.12           0.0%
Prompt-length SJF (heuristic)                         0.43      4.73         -19.8%
LAS / tokens-generated-so-far (no prediction)         0.77     10.83        -311.4%
Oracle SJF (knows output length)                      0.31      2.52          79.4%
Oracle SRPT (knows output length)                     0.29      2.58         100.0%

AzureLLMInferenceTrace_conv.csv  rho=0.9  n=19366
policy                                                mean       p99  gain captured
FCFS                                                  2.42     12.95           0.0%
Prompt-length SJF (heuristic)                         3.19     63.73         -63.7%
LAS / tokens-generated-so-far (no prediction)         6.83    118.36        -362.3%
Oracle SJF (knows output length)                      1.25     20.29          96.0%
Oracle SRPT (knows output length)                     1.20     20.49         100.0%
```

## Sensitivity

Oracle SRPT mean-latency reduction versus FCFS at 90% load, for different prefill costs:

| prefill cost | reduction |
|---|---|
| 0     | 55.7% |
| 0.005 | 53.9% |
| 0.02  | 50.4% |
| 0.05  | 49.8% |

The gain stays in a similar range regardless of how prefill is priced.

## Limitations

Single server, no batching. Preemption is free, with no cost for keeping or recomputing KV cache. Service time is decode tokens plus a prefill cost times prompt tokens.

## Next

I'd extend this to batched serving with KV-cache preemption costs, using a vLLM-based simulator instead of this single-server model. I'd also want to test how much of the oracle gain a learned length predictor recovers against these baselines, given the same signals.

## Data source

Trace data from the Azure Public Dataset (github.com/Azure/AzurePublicDataset).
