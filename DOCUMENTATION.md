# Representation Engineering (RepE) - API Documentation

> **Paper**: [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)

This document provides comprehensive API documentation for the `repe` library.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [API Reference](#api-reference)
   - [Pipeline Registration](#pipeline-registration)
   - [RepReading Pipeline](#repreading-pipeline)
   - [RepControl Pipeline](#repcontrol-pipeline)
   - [RepReader Classes](#repreader-classes)
   - [WrappedReadingVecModel](#wrappedreadingvecmodel)
4. [Data Format](#data-format)
5. [Example Workflows](#example-workflows)

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from repe import repe_pipeline_registry

# Register RepE pipelines with HuggingFace
repe_pipeline_registry()

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                              torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Create pipelines
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline = pipeline("rep-control", model=model, tokenizer=tokenizer, 
                                 layers=list(range(-1, -20, -1)), block_name="decoder_block")
```

---

## Core Concepts

### Representation Reading (RepReading)

RepReading extracts **concept directions** from a model's hidden states. These directions capture high-level concepts like "honesty", "emotion", or "bias" as vectors in the model's representation space.

**Key idea**: Given contrastive pairs of prompts (e.g., "honest" vs "dishonest"), PCA on the differences between their hidden states reveals directions that encode the concept.

### Representation Control (RepControl)

RepControl modifies model behavior by adding or subtracting concept directions to hidden states during generation.

```
modified_hidden = original_hidden + coefficient * concept_direction
```

- Positive coefficient: amplify the concept
- Negative coefficient: suppress the concept

---

## API Reference

### Pipeline Registration

```python
repe_pipeline_registry()
```

Registers `"rep-reading"` and `"rep-control"` task types with HuggingFace Transformers pipelines. **Must be called before creating pipelines.**

---

### RepReading Pipeline

```python
from transformers import pipeline
rep_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
```

#### `get_directions()`

Extract concept directions from contrastive training data.

```python
rep_reader = rep_pipeline.get_directions(
    train_inputs: List[str],              # Contrastive prompts (flattened pairs)
    rep_token: int = -1,                  # Token position for hidden state extraction
    hidden_layers: List[int] = [-1],      # Layer indices (negative = from end)
    n_difference: int = 1,                # Number of pairwise difference operations
    batch_size: int = 8,
    train_labels: List[List[bool]] = None, # For sign determination
    direction_method: str = 'pca',        # 'pca', 'cluster_mean', or 'random'
    direction_finder_kwargs: dict = {},   # e.g., {"n_components": 1}
    which_hidden_states: str = None,      # For encoder-decoder: 'encoder' or 'decoder'
    **tokenizer_args                      # padding, truncation, max_length, etc.
) -> RepReader
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_inputs` | `List[str]` | Flattened list of contrastive prompts: `[pos1, neg1, pos2, neg2, ...]` |
| `rep_token` | `int` | Token position for hidden state extraction. `-1` = last token |
| `hidden_layers` | `List[int]` | Layers to extract. Negative indices count from end |
| `n_difference` | `int` | Number of pairwise difference operations (usually 1) |
| `direction_method` | `str` | `'pca'` (recommended), `'cluster_mean'`, or `'random'` |
| `train_labels` | `List[List[bool]]` | Labels indicating which prompt is "positive" in each pair |

**Returns:** `RepReader` object with extracted directions.

#### `__call__()` (Inference)

Project text onto concept directions.

```python
scores = rep_pipeline(
    inputs: Union[str, List[str]],
    rep_token: int = -1,
    hidden_layers: List[int] = [-1],
    rep_reader: RepReader = None,  # If provided, returns projected scores
    component_index: int = 0,
    batch_size: int = 8,
    **tokenizer_args
) -> Dict[int, np.ndarray]  # layer -> scores
```

---

### RepControl Pipeline

```python
control_pipeline = pipeline(
    "rep-control", 
    model=model, 
    tokenizer=tokenizer,
    layers=list(range(-1, -20, -1)),  # Layers to apply control
    block_name="decoder_block",        # Block to wrap
    control_method="reading_vec"       # Currently only option
)
```

#### `__call__()`

Generate text with concept control.

```python
outputs = control_pipeline(
    text_inputs: Union[str, List[str]],
    activations: Dict[int, torch.Tensor] = None,  # layer -> activation to add
    **generation_kwargs  # max_new_tokens, temperature, etc.
) -> List[Dict]
```

**Creating activations from a RepReader:**

```python
# Get direction and sign
direction = rep_reader.directions[layer]
sign = rep_reader.direction_signs[layer]

# Create activation dict
coeff = 1.5  # strength of control
activations = {}
for layer in layers:
    activations[layer] = coeff * sign * torch.tensor(rep_reader.directions[layer])
```

---

### RepReader Classes

All RepReaders inherit from the abstract `RepReader` base class.

#### `PCARepReader` (Recommended)

```python
from repe.rep_readers import PCARepReader

reader = PCARepReader(n_components=1)
```

Extracts concept directions using PCA on the differences between contrastive hidden states. The first principal component typically captures the concept of interest.

#### `ClusterMeanRepReader`

```python
from repe.rep_readers import ClusterMeanRepReader

reader = ClusterMeanRepReader()
```

Uses the difference between cluster means as the concept direction. Requires `n_difference=1`.

#### `RandomRepReader`

```python
from repe.rep_readers import RandomRepReader

reader = RandomRepReader(needs_hiddens=True)
```

Returns random directions. Useful as a baseline.

#### RepReader Properties

| Property | Type | Description |
|----------|------|-------------|
| `directions` | `Dict[int, np.ndarray]` | Layer -> direction array (n_components, hidden_size) |
| `direction_signs` | `Dict[int, int]` | Layer -> sign (-1 or 1) for interpreting projections |
| `n_components` | `int` | Number of components (directions) per layer |

#### `transform()`

Project hidden states onto concept directions.

```python
scores = rep_reader.transform(
    hidden_states: Dict[int, np.ndarray],  # layer -> (n_examples, hidden_size)
    hidden_layers: List[int],
    component_index: int = 0
) -> Dict[int, np.ndarray]  # layer -> (n_examples,)
```

---

### WrappedReadingVecModel

Low-level API for activation manipulation.

```python
from repe import WrappedReadingVecModel

wrapped_model = WrappedReadingVecModel(model, tokenizer)
```

#### Key Methods

```python
# Wrap specific layers for control
wrapped_model.wrap_block(layer_ids=[10, 15, 20], block_name="decoder_block")

# Set activation controller
wrapped_model.set_controller(
    layer_ids: List[int],
    activations: Dict[int, torch.Tensor],
    block_name: str = 'decoder_block',
    token_pos: int = None,      # Apply to specific token position
    masks: torch.Tensor = None, # Apply with masking
    normalize: bool = False,
    operator: str = 'linear_comb'  # 'linear_comb', 'piecewise_linear', 'projection'
)

# Reset controller
wrapped_model.reset()

# Unwrap all layers
wrapped_model.unwrap()
```

#### Control Operators

| Operator | Formula | Use Case |
|----------|---------|----------|
| `'linear_comb'` | `hidden + activation` | Standard additive control |
| `'piecewise_linear'` | `hidden + max(0, activation)` | One-sided control |
| `'projection'` | `hidden - proj(hidden, activation)` | Remove component |

---

## Data Format

### Contrastive Training Data

Training data should be pairs of prompts that contrast the concept of interest:

```python
# Format: [positive1, negative1, positive2, negative2, ...]
train_data = [
    "[INST] Act as an honest person [/INST] The sky is",
    "[INST] Act as a dishonest person [/INST] The sky is",
    "[INST] Act as an honest person [/INST] Water freezes at",
    "[INST] Act as a dishonest person [/INST] Water freezes at",
    # ...
]

# Labels indicate which is "positive" in each pair (after shuffling)
train_labels = [
    [True, False],   # First prompt is positive
    [False, True],   # Second prompt is positive (after shuffle)
    # ...
]
```

### Template Pattern

```python
user_tag = "[INST]"
assistant_tag = "[/INST]"
template = f"{user_tag} Pretend you're {{type}} person. {assistant_tag} {{statement}}"

honest = template.format(type="an honest", statement="The earth is round")
dishonest = template.format(type="an untruthful", statement="The earth is round")
```

---

## Example Workflows

### 1. Honesty Detection

```python
from repe import repe_pipeline_registry
from transformers import pipeline

repe_pipeline_registry()

# Setup
rep_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

# Create contrastive data
honest_template = "[INST] Pretend you're an honest person. [/INST] "
dishonest_template = "[INST] Pretend you're an untruthful person. [/INST] "
statements = ["The sky is blue", "Water is wet", "Fire is hot"]

train_data = []
for s in statements:
    train_data.extend([honest_template + s, dishonest_template + s])

# Extract honesty direction
rep_reader = rep_pipeline.get_directions(
    train_data,
    hidden_layers=hidden_layers,
    direction_method='pca',
    padding=True,
    truncation=True
)

# Detect honesty in new text
test_text = "[INST] Tell me about the moon. [/INST] The moon is made of cheese."
scores = rep_pipeline(test_text, hidden_layers=hidden_layers, rep_reader=rep_reader)
```

### 2. Emotion Control

```python
# Setup control pipeline
layers = list(range(-5, -20, -1))
control_pipeline = pipeline("rep-control", model=model, tokenizer=tokenizer, layers=layers)

# Create activations from emotion direction
activations = {}
for layer in layers:
    sign = emotion_rep_reader.direction_signs[layer]
    direction = torch.tensor(emotion_rep_reader.directions[layer]).to(model.device)
    activations[layer] = 1.5 * sign * direction  # Amplify happiness

# Generate with emotion control
output = control_pipeline(
    "Tell me about your day.",
    activations=activations,
    max_new_tokens=100,
    do_sample=True
)
```

### 3. Bias Mitigation

```python
# Negative coefficient to suppress bias direction
activations = {}
for layer in layers:
    sign = bias_rep_reader.direction_signs[layer]
    direction = torch.tensor(bias_rep_reader.directions[layer]).to(model.device)
    activations[layer] = -1.0 * sign * direction  # Reduce bias

output = control_pipeline(prompt, activations=activations, max_new_tokens=100)
```

---

## Supported Models

RepE has been tested with:

- **LLaMA 2** (7B, 13B, 70B) - Chat and base variants
- **Mistral** (7B)
- Other decoder-only transformers should work with `block_name="decoder_block"`

For encoder-decoder models, use `which_hidden_states='encoder'` or `'decoder'`.

---

## Tips and Best Practices

1. **Use multiple layers**: Concept information is distributed across layers. Extract directions from many layers and use the best-performing one.

2. **Validate with held-out data**: Check that your extracted direction generalizes by testing on held-out examples.

3. **Start with small coefficients**: When using RepControl, start with small activation coefficients (0.5-1.5) and increase gradually.

4. **Include diverse examples**: More diverse training examples = better concept generalization.

5. **Use appropriate token position**: `-1` (last token) works well for causal LMs taking the completion direction.
