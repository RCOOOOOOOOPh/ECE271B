# ECE271B

this is UCSD ECE271B 24WI project about LoRA, VeRA and DoRA

translate_ft.ipynb, translate_lora.ipynb, translate_vera.ipynb and translate_dora.ipynb use my own code for lora, vera and dora

translate_peft_lora.ipynb and translate_peft_dora.ipynb use the peft module by huggingface

gpt2_generation.ipynb is GPT2 finetune in E2E NLG benchmark, gpt2_generation.ipynb is the same task using lora from peft.

our accuracies:

|                                                | finetune | lora   | dora   |
| ---------------------------------------------- | -------- | ------ | ------ |
| parameter                                      | 77943296 | 589824 | 608256 |
| parameter percent (compared to full  finetune) | 100%     | 0.76%  | 0.78%  |
| bleu1                                          | 0.5918   | 0.5746 | 0.5739 |
| bleu2                                          | 0.4574   | 0.4317 | 0.4328 |
| bleu3                                          | 0.3603   | 0.3337 | 0.3359 |
| bleu4                                          | 0.29     | 0.2616 | 0.2654 |
| meteor                                         | 0.3392   | 0.3303 | 0.328  |
| rouge_l                                        | 0.5421   | 0.5201 | 0.5203 |
| cider_d                                        | 2.7158   | 2.4851 | 2.4544 |
| spice                                          | 0.4726   | 0.4451 | 0.4432 |
| spider                                         | 1.5942   | 1.4651 | 1.4488 |

Translate Examples:

1\.Hello, World (你好，世界)

ft: How do you do, the world?

lora: Hello, world.

dora: Hello, world.

2\. I love you, You love me, Sweet Snow Ice City (an icecream and milk tea store name) is so sweet. （你爱我我爱你，蜜雪冰城甜蜜蜜）

ft: You love me. I love you, honey.

lora: You love me. I love you, honey.

dora: You love me. I love you, honey.

3\. Man was born free, and everywhere he is in chains. (人是生而自由的，却处处受到束缚。)

ft: Man is born free, but bound everywhere.

lora: Man is born free and bound everywhere.

dora: Man is born free and bound everywhere.

4\. Honesty and diligence should be your eternal mates. (诚实与勤勉应该成为你永久的伴侣。)

ft: honesty and hard work should be your permanent mates.

lora: Integrity and diligence should be your permanent companions.

dora: Integrity and diligence should be your permanent companions.

