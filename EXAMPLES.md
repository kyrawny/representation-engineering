# Representation Engineering - Usage Examples

This guide provides practical examples for common RepE workflows.

---

## Table of Contents

1. [Setup](#setup)
2. [Basic Examples](#basic-examples)
   - [Extract a Concept Direction](#example-1-extract-a-concept-direction)
   - [Detect Concepts in Text](#example-2-detect-concepts-in-text)
   - [Control Generation](#example-3-control-generation)
3. [Application Examples](#application-examples)
   - [Honesty Detection (Lie Detector)](#honesty-detection)
   - [Emotion Control](#emotion-control)
   - [Bias Mitigation](#bias-mitigation)
   - [Memorization Reduction](#memorization-reduction)
4. [Creating Custom Concepts](#creating-custom-concepts)

---

## Setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from repe import repe_pipeline_registry

# Register pipelines (required once)
repe_pipeline_registry()

# Load model
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Model-specific prompt format
USER_TAG = "[INST]"
ASSISTANT_TAG = "[/INST]"
```

---

## Basic Examples

### Example 1: Extract a Concept Direction

```python
# Create pipeline
rep_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
hidden_layers = list(range(-1, -20, -1))  # Last 19 layers

# Define contrastive prompts (concept: confidence)
positive_template = f"{USER_TAG} Act as a confident person. {ASSISTANT_TAG} "
negative_template = f"{USER_TAG} Act as an uncertain person. {ASSISTANT_TAG} "

statements = [
    "I believe that",
    "The answer is",
    "In my opinion",
    "I think that",
]

# Create training data (alternating positive/negative)
train_data = []
for s in statements:
    train_data.extend([positive_template + s, negative_template + s])

# Extract direction
rep_reader = rep_pipeline.get_directions(
    train_data,
    hidden_layers=hidden_layers,
    direction_method='pca',
    n_difference=1,
    padding=True,
    truncation=True,
    max_length=256
)

print(f"Extracted {len(rep_reader.directions)} layer directions")
print(f"Direction shape: {rep_reader.directions[-1].shape}")
```

### Example 2: Detect Concepts in Text

```python
# Test on new text
test_prompts = [
    f"{USER_TAG} How do you feel about this? {ASSISTANT_TAG} I'm absolutely certain this is correct.",
    f"{USER_TAG} How do you feel about this? {ASSISTANT_TAG} I'm not really sure about this.",
]

# Get projection scores
scores = rep_pipeline(
    test_prompts, 
    hidden_layers=hidden_layers, 
    rep_reader=rep_reader,
    padding=True,
    truncation=True
)

# Analyze scores
for i, prompt in enumerate(test_prompts):
    layer_scores = [scores[i][layer] for layer in hidden_layers]
    avg_score = sum(layer_scores) / len(layer_scores)
    print(f"Prompt {i+1}: score={avg_score:.3f}")
```

### Example 3: Control Generation

```python
# Create control pipeline
layers = list(range(-5, -20, -1))  # Middle-to-late layers
control_pipeline = pipeline(
    "rep-control", 
    model=model, 
    tokenizer=tokenizer, 
    layers=layers,
    block_name="decoder_block"
)

# Create activations from concept direction
def make_activations(rep_reader, layers, coefficient=1.0):
    activations = {}
    for layer in layers:
        sign = rep_reader.direction_signs[layer]
        direction = torch.tensor(rep_reader.directions[layer]).to(model.device)
        direction = direction.to(model.dtype)
        activations[layer] = coefficient * sign * direction
    return activations

# Generate with INCREASED confidence
prompt = f"{USER_TAG} Give me advice about investing. {ASSISTANT_TAG}"

confident_output = control_pipeline(
    prompt,
    activations=make_activations(rep_reader, layers, coefficient=1.5),
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

# Generate with DECREASED confidence
uncertain_output = control_pipeline(
    prompt,
    activations=make_activations(rep_reader, layers, coefficient=-1.5),
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

print("Confident:", confident_output[0]['generated_text'])
print("\nUncertain:", uncertain_output[0]['generated_text'])
```

---

## Application Examples

### Honesty Detection

Detect when a model is being truthful vs. deceptive.

```python
import pandas as pd
import numpy as np

# Load facts dataset
df = pd.read_csv("data/facts/facts_true_false.csv")
true_statements = df[df['label'] == 1]['statement'].tolist()[:100]

# Create honesty training data
honest_template = f"{USER_TAG} Pretend you're an honest person making statements. {ASSISTANT_TAG} "
dishonest_template = f"{USER_TAG} Pretend you're an untruthful person making statements. {ASSISTANT_TAG} "

train_data = []
train_labels = []
for stmt in true_statements[:50]:
    pair = [honest_template + stmt, dishonest_template + stmt]
    np.random.shuffle(pair)
    train_data.extend(pair)
    train_labels.append([pair[0].startswith(honest_template), pair[1].startswith(honest_template)])

# Extract honesty direction
honesty_reader = rep_pipeline.get_directions(
    train_data,
    hidden_layers=hidden_layers,
    train_labels=train_labels,
    direction_method='pca',
    padding=True,
    truncation=True
)

# Test lie detection
def is_honest(text, threshold=0.0):
    scores = rep_pipeline([text], hidden_layers=hidden_layers, rep_reader=honesty_reader)
    avg_score = np.mean([scores[0][layer] for layer in hidden_layers])
    return avg_score > threshold

# Example usage
print(is_honest(f"{ASSISTANT_TAG} The Earth orbits the Sun."))  # True
print(is_honest(f"{ASSISTANT_TAG} The moon is made of cheese."))  # Likely False
```

### Emotion Control

Control the emotional tone of generated responses.

```python
import json
import os

# Define emotion contrasts
emotions = {
    "happiness": (["joyful", "happy", "cheerful"], ["dejected", "unhappy", "sad"]),
    "anger": (["angry", "furious", "irritated"], ["calm", "peaceful", "pleased"]),
    "sadness": (["sad", "depressed", "miserable"], ["cheerful", "happy", "optimistic"]),
}

def create_emotion_data(emotion, positive_adj, negative_adj, scenarios):
    train_data = []
    for scenario in scenarios:
        pos_adj = np.random.choice(positive_adj)
        neg_adj = np.random.choice(negative_adj)
        train_data.extend([
            f"{USER_TAG} Act as if you are extremely {pos_adj}. {ASSISTANT_TAG} {scenario}",
            f"{USER_TAG} Act as if you are extremely {neg_adj}. {ASSISTANT_TAG} {scenario}",
        ])
    return train_data

# Sample scenarios
scenarios = [
    "Today I woke up and",
    "When I think about the future,",
    "Looking at the world around me,",
    "In this moment I feel",
]

# Extract emotion directions
emotion_readers = {}
for emotion, (pos_adj, neg_adj) in emotions.items():
    train_data = create_emotion_data(emotion, pos_adj, neg_adj, scenarios * 10)
    emotion_readers[emotion] = rep_pipeline.get_directions(
        train_data,
        hidden_layers=hidden_layers,
        direction_method='pca',
        padding=True,
        truncation=True
    )

# Generate with increased happiness
prompt = f"{USER_TAG} Tell me about your day. {ASSISTANT_TAG}"
happy_activations = make_activations(emotion_readers["happiness"], layers, coefficient=2.0)

happy_output = control_pipeline(
    prompt,
    activations=happy_activations,
    max_new_tokens=100,
    do_sample=True
)
print("Happy response:", happy_output[0]['generated_text'])
```

### Bias Mitigation

Reduce biased outputs by steering away from stereotype directions.

```python
from datasets import load_dataset

# Load StereoSet dataset
dataset = load_dataset('stereoset', 'intrasentence')['validation']
race_data = [d for d in dataset if d['bias_type'] == 'race']

# Create training data
template = f"{USER_TAG} Consider the bias of the following scenario.\nScenario: {{scenario}} {ASSISTANT_TAG} "

train_data = []
for item in race_data[:100]:
    sentences = item['sentences']
    anti_idx = sentences['gold_label'].index(0)  # Anti-stereotype
    stereo_idx = sentences['gold_label'].index(1)  # Stereotype
    
    train_data.extend([
        template.format(scenario=sentences['sentence'][anti_idx]),
        template.format(scenario=sentences['sentence'][stereo_idx]),
    ])

# Extract bias direction
bias_reader = rep_pipeline.get_directions(
    train_data,
    hidden_layers=hidden_layers,
    direction_method='pca',
    padding=True,
    truncation=True
)

# Generate with REDUCED bias (negative coefficient)
prompt = f"{USER_TAG} Write a story about a doctor. {ASSISTANT_TAG}"
unbiased_output = control_pipeline(
    prompt,
    activations=make_activations(bias_reader, layers, coefficient=-1.5),
    max_new_tokens=150,
    do_sample=True
)
```

### Memorization Reduction

Reduce verbatim recall of memorized content.

```python
# Load memorization data (literary openings)
with open("data/memorization/literary_openings/real.json") as f:
    seen_texts = json.load(f)
with open("data/memorization/literary_openings/fake.json") as f:
    unseen_texts = json.load(f)

# Create training data: seen vs unseen
train_data = []
for seen, unseen in zip(seen_texts[:50], unseen_texts[:50]):
    train_data.extend([seen, unseen])

# Extract memorization direction
mem_reader = rep_pipeline.get_directions(
    train_data,
    hidden_layers=hidden_layers,
    direction_method='pca',
    padding=True,
    truncation=True
)

# Generate with reduced memorization
prompt = "It was the best of times, it was the"  # Famous opening

# Normal generation
normal_output = control_pipeline(prompt, max_new_tokens=50)

# With memorization reduction
reduced_output = control_pipeline(
    prompt,
    activations=make_activations(mem_reader, layers, coefficient=-2.0),
    max_new_tokens=50
)

print("Normal:", normal_output[0]['generated_text'])
print("Reduced memorization:", reduced_output[0]['generated_text'])
```

---

## Creating Custom Concepts

To create a direction for any concept:

### Step 1: Define Contrastive Prompts

```python
# Template: What makes your concept present vs absent?
concept = "formality"
positive_template = f"{USER_TAG} Respond in a very formal, professional manner. {ASSISTANT_TAG} "
negative_template = f"{USER_TAG} Respond in a casual, informal way. {ASSISTANT_TAG} "
```

### Step 2: Create Diverse Examples

```python
scenarios = [
    "The meeting has been rescheduled to",
    "I wanted to let you know that",
    "Regarding your request,",
    "Thank you for",
    # Add 50+ diverse examples for best results
]

train_data = []
for s in scenarios:
    train_data.extend([positive_template + s, negative_template + s])
```

### Step 3: Extract and Validate

```python
# Extract
custom_reader = rep_pipeline.get_directions(
    train_data,
    hidden_layers=hidden_layers,
    direction_method='pca',
    padding=True,
    truncation=True
)

# Validate on held-out examples
test_formal = f"{ASSISTANT_TAG} I hereby acknowledge receipt of your correspondence."
test_casual = f"{ASSISTANT_TAG} Hey! Got your message, thanks!"

formal_score = rep_pipeline([test_formal], hidden_layers=hidden_layers, rep_reader=custom_reader)
casual_score = rep_pipeline([test_casual], hidden_layers=hidden_layers, rep_reader=custom_reader)

# Formal should score higher than casual
print(f"Formal: {np.mean([formal_score[0][l] for l in hidden_layers]):.3f}")
print(f"Casual: {np.mean([casual_score[0][l] for l in hidden_layers]):.3f}")
```

### Step 4: Apply Control

```python
# Use for generation control
formal_activations = make_activations(custom_reader, layers, coefficient=1.5)
casual_activations = make_activations(custom_reader, layers, coefficient=-1.5)
```

---

## Saving and Loading Directions

```python
import pickle

# Save
with open("my_concept_reader.pkl", "wb") as f:
    pickle.dump(rep_reader, f)

# Load
with open("my_concept_reader.pkl", "rb") as f:
    loaded_reader = pickle.load(f)
```

---

## Troubleshooting

### Common Issues

1. **"Tensors with different lengths" error**
   - Add `padding=True, truncation=True` to `get_directions()` and pipeline calls

2. **Poor concept separation**
   - Use more diverse training examples (50+)
   - Try different layers - information varies by layer
   - Increase contrast between positive/negative prompts

3. **Generation unchanged with control**
   - Increase coefficient (try 2.0-3.0)
   - Check that layers include mid-to-late layers (-10 to -20)
   - Verify direction signs are applied correctly

4. **CUDA out of memory**
   - Reduce batch_size in `get_directions()`
   - Use fewer layers
   - Use smaller model quantization
