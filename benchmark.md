```bash
python3 benchmark_compare.py \
  --impl compare \
  --model-path /gz-data/model/Qwen3-4B/ \
  --workload online \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-chunk-tokens 256 \
  --max-num-seqs 128 \
  --short-input-len-min 32 \
  --short-input-len-max 128 \
  --long-input-len-min 1024 \
  --long-input-len-max 2048 \
  --min-output-len 64 \
  --max-output-len 128 \
  --online-initial-long-requests 16 \
  --online-late-short-requests 64 \
  --online-start-gap-ms 100 \
  --online-arrival-interval-ms 50 \
  --repeat 5 \
  --aggregation median \
  --json-out artifacts/online_compare_repeat5.json \
  --csv-out artifacts/online_compare_repeat5.csv \
  --markdown-out artifacts/online_compare_repeat5.md

```

# Benchmark Comparison

- mode: `compare`
- workload: `online`
- num_seqs: `80`
- repeat_count: `5`
- aggregation: `median`

| Metric                         |        original |         current |    Change |
| ------------------------------ | --------------: | --------------: | --------: |
| `decode_tok_per_s`             | 1246.9552 tok/s | 1256.2294 tok/s |   0.7437% |
| `late_short_latency_p50_s`     |         3.8577s |         3.8095s |   1.2486% |
| `late_short_latency_p95_s`     |         5.6823s |         5.6316s |   0.8928% |
| `late_short_queue_delay_p50_s` |         0.1492s |         0.1657s | -11.0732% |
| `late_short_queue_delay_p95_s` |         0.3076s |         0.3724s | -21.0988% |
| `late_short_ttft_p50_s`        |         1.3462s |         1.1636s |  13.5663% |
| `late_short_ttft_p95_s`        |         2.4668s |         2.1167s |  14.1927% |
| `latency_all_p50_s`            |         4.6245s |         4.5525s |   1.5565% |
| `latency_all_p95_s`            |         6.0602s |         6.0229s |   0.6144% |
| `prompt_tok_per_s`             | 4888.1985 tok/s | 4924.5544 tok/s |   0.7437% |
| `queue_delay_all_p50_s`        |         0.1044s |         0.1157s | -10.8594% |
| `queue_delay_all_p95_s`        |         0.3040s |         0.3629s | -19.3920% |
| `request_per_s`                |   12.7926 req/s |   12.8877 req/s |   0.7437% |
| `total_tok_per_s`              | 6135.1537 tok/s | 6180.7838 tok/s |   0.7437% |
| `ttft_all_p50_s`               |         1.3462s |         1.5065s | -11.9048% |
| `ttft_all_p95_s`               |         2.4944s |         3.0151s | -20.8766% |
这组 5 次重复取中位数的结果就比较能拿出去说了，而且结论比单次更稳。

可以下这个结论：你的优化更偏“改善被长请求压住的短请求体验”，不是“所有请求整体都更快”。最值得讲的是：

- late_short_ttft_p50 从 1.3462s 降到 1.1636s，下降 13.6%
- late_short_ttft_p95 从 2.4668s 降到 2.1167s，下降 14.2%
- total_tok_per_s 提升 0.74%
- decode_tok_per_s 提升 0.74%

这说明 chunked prefill + prefill/decode 混合调度 确实缓解了长 prompt 对晚到短请求的阻塞，而且没有明显牺牲吞吐。  
但也要注意，ttft_all_p50/p95 是变差的，所以不要写“整体 TTFT 优化”，要写“短请求交互时延优化”更准确。queue_delay 这组数据更适合内部分析，不建议作为简历主指标。


nano-vllm 推理引擎中实现 chunked prefill 与 Prefill/Decode 混合调度；在在线混合负载压测下（5 次重复取中位数），晚到短请求 TTFT P50/P95 分别降低 13.6%/14.2%，同时总吞吐提升 0.74%，验证了该调度策略在缓解长请求阻塞、优化交互时延方面的效果。
