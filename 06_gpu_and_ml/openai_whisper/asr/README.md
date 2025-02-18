To deploy model

```
modal deploy model.py
```

To test latency, update the URL in `bench.py` and run

```
pytest bench.py
```

------------------------

For just benchmarking modal latency

```
modal deploy test_crazy.py
```

```
python test_script.py
```



------------------------
Hugging Face benchmark V3 vs Turbo:
```
modal run hf_v3_vs_turbo_baseline.py
```
