# Affect Control Theory (ACT) - Representation Engineering Example

This example demonstrates how to use Representation Engineering to extract, read, and control the **Evaluation**, **Potency**, and **Activity** (EPA) dimensions from David Heise's *Affect Control Theory* (2007).

## EPA Framework

| Dimension | Positive | Negative | Description |
|-----------|----------|----------|-------------|
| **Evaluation** | Good | Bad | Morality, altruism, social desirability |
| **Potency** | Potent | Impotent | Power, authority, dominance, strength |
| **Activity** | Active | Inactive | Energy level, speed, volatility, liveliness |

## Notebooks

### 1. `act_epa_extraction.ipynb`
Extract EPA direction vectors using contrastive prompts with Llama-3.1-8B-Instruct.
- Creates contrastive pairs for each EPA dimension
- Uses PCA to extract directions from hidden states
- Saves directions for later use

### 2. `act_reading_control.ipynb`
Read EPA from user input and control response EPA.
- **Reading**: Format user input as assistant output to read EPA values
- **Control**: Apply EPA vectors to steer generation
- Demonstrates all three operators: Linear Combination, Piecewise, Projection

### 3. `act_visualizations.ipynb`
Visualizations of EPA representations:
- **t-SNE**: Cluster visualization in early/late layers (Figure 14 style)
- **LAT Scans**: Temporal activation across layers (Figure 8 style)
- **Per-Token**: Token-level EPA intensity (Figure 9 style)

## Model

Uses [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

**Prompt Template:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_response}
```

## References

- Heise, D. R. (2007). *Expressive Order: Confirming Sentiments in Social Actions*. Springer.
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405.
