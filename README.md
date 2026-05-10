# Parameter-Efficient Fine-Tuning with LoRA Optimizers

This project explores different optimizers for parameter-efficient fine-tuning using LoRA. Instead of fine-tuning all the parameters of a transformer model, we freeze most of the model and train only a small number of LoRA adapter parameters.

We use `distilbert-base-uncased` and run experiments on GLUE tasks such as SST-2, MRPC, and RTE.

## Project Goal

The main goal of this project is to understand whether more advanced optimizers help when we are only training LoRA adapters.

In particular, we compare standard first-order optimizers with a K-FAC-style optimizer. K-FAC uses curvature information, so in theory it could help optimization. However, our experiments show that a simpler diagonal adaptive optimizer works better in this setting.

## Optimizers Compared

| Optimizer | Type | Description |
|---|---|---|
| AdamW | First-order | Standard optimizer commonly used for transformers |
| SGDM | First-order | SGD with momentum |
| adapter_diag | First-order adaptive | Uses a diagonal running estimate of squared gradients for LoRA parameters |
| adapter_kfac | Approximate second-order | Uses K-FAC-style Fisher preconditioning for LoRA parameters |

The `adapter_diag` optimizer is still first-order because it only uses gradients. It does not compute Hessians, Fisher matrices, or full curvature information.

## Model Setup

- Model: `distilbert-base-uncased`
- Fine-tuning method: LoRA
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.1`
- Target modules: `q_lin` and `v_lin`
- Trainable parameters: `739,586`
- Total parameters: `67,694,596`
- Trainable percentage: about `1.09%`

This means we fine-tune only a small part of the model while keeping most of DistilBERT frozen.

## Results

Current single-seed results:

| Task | Optimizer | Best Accuracy | Best F1 |
|---|---|---:|---:|
| SST-2 | AdamW | 89.91 | n/a |
| SST-2 | adapter_diag | 89.91 | n/a |
| SST-2 | SGDM | 87.50 | n/a |
| SST-2 | adapter_kfac | 84.86 | n/a |
| MRPC | AdamW | 83.58 | 88.62 |
| MRPC | adapter_diag | 84.80 | 89.23 |
| RTE | AdamW | 60.29 | n/a |
| RTE | adapter_diag | 63.54 | n/a |

The best overall result came from `adapter_diag`. It matched AdamW on SST-2 and performed better than AdamW on MRPC and RTE.

K-FAC was stable and the validation accuracy improved during training, but it did not outperform the simpler first-order methods in our current experiments.

## Main Takeaway

The most interesting result is that the more complex optimizer was not the best one.

Even though K-FAC uses curvature information, the simpler `adapter_diag` optimizer worked better for our LoRA setup. This suggests that when the number of trainable parameters is already very small, a simple adaptive first-order method may be enough.

Our conclusion is not that K-FAC never works for PEFT. Instead, based on our experiments, direct K-FAC-style optimization did not give an advantage over a simpler diagonal adaptive optimizer.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run SST-2:

```bash
python train.py --config configs/sst2.yaml
```

Run MRPC:

```bash
python train.py --config configs/mrpc.yaml
```

Run RTE:

```bash
python train.py --config configs/rte.yaml
```

## Repository Structure

```text
configs/           Experiment configuration files
src/data.py        Loads and tokenizes GLUE datasets
src/model.py       Builds DistilBERT with LoRA adapters
src/optimizers.py  Contains AdamW, SGDM, adapter_diag, and adapter_kfac
src/kfac.py        K-FAC integration using ASDL
src/trainer.py     Training and evaluation loop
outputs/           Saved experiment histories and adapter checkpoints
paper/             Manuscript draft files
train.py           Main script for running experiments
```

## Paper Draft

The draft paper is available in the `paper/` folder:

```text
paper/manuscript.pdf
paper/manuscript.docx
paper/manuscript.md
```

## Future Work

Some things we would like to add next:

- Run each experiment with multiple random seeds
- Add more complete K-FAC hyperparameter sweep results
- Run K-FAC on MRPC and RTE
- Compare memory usage and training time more carefully
- Try larger models such as BERT-base or RoBERTa-base
- Compare with more LoRA-specific optimizers such as LoRA+

## Summary

This project shows that optimizer choice matters for LoRA fine-tuning. In our experiments, a simple diagonal adaptive optimizer performed better than the K-FAC-style optimizer, even though K-FAC uses more advanced curvature information.
