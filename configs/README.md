# configs/

`hindi/model.json` and `nepali/model.json` hold both the architecture (`vocab_size`, `d_model`, `num_heads`, `num_layers`, `d_ff`, `seq_len`,
`dropout`) and the training hyper-parameters (batch, accumulation, learning rate, warmup, weight decay, gradient clip, token budgets, logging cadence).
The two files are identical except for `language`.

Note: the learning-rate floor used by the code (10 % of peak) differs from the schedule visible in the released Hindi log (decay to zero);
see [docs/04](../docs/04-pretraining.md).
