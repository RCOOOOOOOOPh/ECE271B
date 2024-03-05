'''from aac_metrics import evaluate

candidates: list[str] = ["a man is speaking", "rain falls"]
mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]

corpus_scores, _ = evaluate(candidates, mult_references)
print(corpus_scores)'''
import torch
from torch import tensor
sb = {'bleu_1': tensor(0.5739, dtype=torch.float64), 'bleu_2': tensor(0.4328, dtype=torch.float64), 'bleu_3': tensor(0.3359, dtype=torch.float64), 'bleu_4': tensor(0.2654, dtype=torch.float64), 'meteor': tensor(0.3280, dtype=torch.float64), 'rouge_l': tensor(0.5203, dtype=torch.float64), 'cider_d': tensor(2.4544, dtype=torch.float64), 'spice': tensor(0.4432, dtype=torch.float64), 'spider': tensor(1.4488, dtype=torch.float64)}
for k, v in sb.items():
    print(k, float(v))