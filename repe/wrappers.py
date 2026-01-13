"""
High-level wrapper classes for common representation engineering workflows.

These wrappers provide simplified interfaces for:
- Extracting concept directions from contrastive data
- Detecting concepts in text
- Controlling generation with concept vectors
"""

import numpy as np
import torch
import pickle
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from transformers import pipeline as hf_pipeline

from .rep_readers import RepReader, PCARepReader
from .rep_reading_pipeline import RepReadingPipeline
from .rep_control_pipeline import RepControlPipeline


class ConceptDirection:
    """
    Container for extracted concept direction with metadata.
    
    Attributes:
        name: Human-readable name for the concept.
        directions: Dict mapping layer index to direction array.
        signs: Dict mapping layer index to sign (-1 or 1).
        layers: List of layer indices.
        metadata: Additional metadata dict.
    """
    
    def __init__(
        self,
        name: str,
        rep_reader: RepReader,
        layers: List[int],
        metadata: Optional[Dict] = None,
    ):
        self.name = name
        self.directions = rep_reader.directions
        self.signs = rep_reader.direction_signs
        self.layers = layers
        self.metadata = metadata or {}
        self._rep_reader = rep_reader
    
    def get_activation(
        self,
        layer: int,
        coefficient: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get activation tensor for a specific layer."""
        sign = self.signs.get(layer, 1)
        direction = torch.tensor(self.directions[layer], dtype=dtype)
        if device is not None:
            direction = direction.to(device)
        return coefficient * sign * direction
    
    def get_activations(
        self,
        layers: Optional[List[int]] = None,
        coefficient: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[int, torch.Tensor]:
        """Get activation dict for multiple layers."""
        layers = layers or self.layers
        return {
            layer: self.get_activation(layer, coefficient, device, dtype)
            for layer in layers
            if layer in self.directions
        }
    
    def save(self, path: Union[str, Path]):
        """Save concept direction to file."""
        data = {
            'name': self.name,
            'directions': self.directions,
            'signs': self.signs,
            'layers': self.layers,
            'metadata': self.metadata,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ConceptDirection':
        """Load concept direction from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct RepReader
        rep_reader = PCARepReader()
        rep_reader.directions = data['directions']
        rep_reader.direction_signs = data['signs']
        
        return cls(
            name=data['name'],
            rep_reader=rep_reader,
            layers=data['layers'],
            metadata=data.get('metadata', {}),
        )


class ConceptExtractor:
    """
    High-level wrapper for extracting and detecting concept directions.
    
    Example:
        >>> extractor = ConceptExtractor(model, tokenizer, "honesty")
        >>> concept = extractor.extract_from_pairs(honest_prompts, dishonest_prompts)
        >>> scores = extractor.detect("The sky is blue.")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        concept_name: str = "concept",
        layers: Optional[List[int]] = None,
    ):
        """
        Args:
            model: HuggingFace model.
            tokenizer: HuggingFace tokenizer.
            concept_name: Name for the concept being extracted.
            layers: Layers to extract from. Default: all but first few.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.concept_name = concept_name
        
        # Create pipeline
        self.pipeline = hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)
        
        # Set default layers
        if layers is None:
            n_layers = model.config.num_hidden_layers
            layers = list(range(-1, -min(n_layers - 1, 20), -1))
        self.layers = layers
        
        self._concept: Optional[ConceptDirection] = None
    
    def extract_from_pairs(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        method: str = 'pca',
        n_components: int = 1,
        batch_size: int = 8,
        include_labels: bool = True,
        **tokenizer_kwargs,
    ) -> ConceptDirection:
        """
        Extract concept direction from contrastive prompt pairs.
        
        Args:
            positive_prompts: Prompts representing presence of concept.
            negative_prompts: Prompts representing absence of concept.
            method: Direction extraction method ('pca', 'cluster_mean').
            n_components: Number of PCA components.
            batch_size: Batch size for processing.
            include_labels: Whether to track labels for sign determination.
            **tokenizer_kwargs: Padding, truncation, max_length, etc.
        
        Returns:
            ConceptDirection with extracted directions.
        """
        import random
        
        # Create training data
        train_data = []
        train_labels = []
        
        for pos, neg in zip(positive_prompts, negative_prompts):
            pair = [pos, neg]
            if include_labels:
                is_first_pos = True
                random.shuffle(pair)
                train_labels.append([pair[0] == pos, pair[1] == pos])
            train_data.extend(pair)
        
        # Set default tokenizer kwargs
        tokenizer_kwargs.setdefault('padding', True)
        tokenizer_kwargs.setdefault('truncation', True)
        
        # Extract directions
        rep_reader = self.pipeline.get_directions(
            train_data,
            hidden_layers=self.layers,
            direction_method=method,
            direction_finder_kwargs={'n_components': n_components},
            train_labels=train_labels if include_labels else None,
            batch_size=batch_size,
            **tokenizer_kwargs,
        )
        
        # Create concept direction
        self._concept = ConceptDirection(
            name=self.concept_name,
            rep_reader=rep_reader,
            layers=self.layers,
            metadata={
                'method': method,
                'n_positive': len(positive_prompts),
                'n_negative': len(negative_prompts),
            },
        )
        
        return self._concept
    
    def extract_from_data(
        self,
        train_data: List[str],
        train_labels: Optional[List[List[bool]]] = None,
        method: str = 'pca',
        n_difference: int = 1,
        batch_size: int = 8,
        **tokenizer_kwargs,
    ) -> ConceptDirection:
        """
        Extract concept direction from pre-formatted training data.
        
        Args:
            train_data: Flattened list of contrastive prompts.
            train_labels: Optional labels for sign determination.
            method: Direction extraction method.
            n_difference: Number of pairwise differences.
            batch_size: Batch size for processing.
            **tokenizer_kwargs: Padding, truncation, max_length, etc.
        
        Returns:
            ConceptDirection with extracted directions.
        """
        tokenizer_kwargs.setdefault('padding', True)
        tokenizer_kwargs.setdefault('truncation', True)
        
        rep_reader = self.pipeline.get_directions(
            train_data,
            hidden_layers=self.layers,
            direction_method=method,
            train_labels=train_labels,
            n_difference=n_difference,
            batch_size=batch_size,
            **tokenizer_kwargs,
        )
        
        self._concept = ConceptDirection(
            name=self.concept_name,
            rep_reader=rep_reader,
            layers=self.layers,
            metadata={'method': method},
        )
        
        return self._concept
    
    def detect(
        self,
        text: Union[str, List[str]],
        component_index: int = 0,
        batch_size: int = 8,
        **tokenizer_kwargs,
    ) -> Dict[int, np.ndarray]:
        """
        Detect concept presence in text.
        
        Args:
            text: Text or list of texts to analyze.
            component_index: Which component to use (for multi-component extraction).
            batch_size: Batch size for processing.
            **tokenizer_kwargs: Tokenizer arguments.
        
        Returns:
            Dict mapping layer index to projection scores.
        """
        if self._concept is None:
            raise ValueError("No concept extracted. Call extract_from_pairs() first.")
        
        if isinstance(text, str):
            text = [text]
        
        tokenizer_kwargs.setdefault('padding', True)
        tokenizer_kwargs.setdefault('truncation', True)
        
        scores = self.pipeline(
            text,
            hidden_layers=self.layers,
            rep_reader=self._concept._rep_reader,
            component_index=component_index,
            batch_size=batch_size,
            **tokenizer_kwargs,
        )
        
        # Aggregate if single text
        if len(text) == 1:
            return {layer: scores[0][layer] for layer in self.layers}
        
        # Return per-text scores
        return scores
    
    def get_average_score(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        **kwargs,
    ) -> float:
        """Get average detection score across layers."""
        layers = layers or self.layers
        scores = self.detect(text, **kwargs)
        layer_scores = [float(scores[layer]) for layer in layers if layer in scores]
        return np.mean(layer_scores) if layer_scores else 0.0
    
    def is_concept_present(
        self,
        text: str,
        threshold: float = 0.0,
        **kwargs,
    ) -> bool:
        """Check if concept is present (score above threshold)."""
        return self.get_average_score(text, **kwargs) > threshold
    
    @property
    def concept(self) -> Optional[ConceptDirection]:
        """Get the current concept direction."""
        return self._concept
    
    def set_concept(self, concept: ConceptDirection):
        """Set a pre-extracted concept direction."""
        self._concept = concept
        self.layers = concept.layers
    
    def save(self, path: Union[str, Path]):
        """Save the current concept to file."""
        if self._concept is None:
            raise ValueError("No concept to save.")
        self._concept.save(path)
    
    def load(self, path: Union[str, Path]):
        """Load a concept from file."""
        self._concept = ConceptDirection.load(path)
        self.layers = self._concept.layers


class ConceptController:
    """
    High-level wrapper for controlled generation using concept directions.
    
    Example:
        >>> controller = ConceptController(model, tokenizer)
        >>> controller.add_concept(honesty_concept, strength=1.5)
        >>> output = controller.generate("Tell me about...")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        layers: Optional[List[int]] = None,
        block_name: str = "decoder_block",
    ):
        """
        Args:
            model: HuggingFace model.
            tokenizer: HuggingFace tokenizer.
            layers: Layers to apply control. Default: mid-to-late layers.
            block_name: Name of the block to wrap.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.dtype = next(model.parameters()).dtype
        
        # Set default layers (mid-to-late layers work best for control)
        if layers is None:
            n_layers = model.config.num_hidden_layers
            layers = list(range(-5, -min(n_layers - 1, 20), -1))
        self.layers = layers
        
        # Create control pipeline
        self.pipeline = hf_pipeline(
            "rep-control",
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            block_name=block_name,
        )
        
        # Active concepts
        self._active_concepts: Dict[str, Tuple[ConceptDirection, float]] = {}
    
    def add_concept(self, concept: ConceptDirection, strength: float = 1.0):
        """
        Add a concept to control during generation.
        
        Args:
            concept: ConceptDirection to apply.
            strength: Coefficient (positive = amplify, negative = suppress).
        """
        self._active_concepts[concept.name] = (concept, strength)
    
    def remove_concept(self, concept_name: str):
        """Remove a concept from active control."""
        if concept_name in self._active_concepts:
            del self._active_concepts[concept_name]
    
    def clear_concepts(self):
        """Remove all active concepts."""
        self._active_concepts.clear()
    
    def set_strength(self, concept_name: str, strength: float):
        """Update the strength of an active concept."""
        if concept_name in self._active_concepts:
            concept, _ = self._active_concepts[concept_name]
            self._active_concepts[concept_name] = (concept, strength)
    
    def _compute_activations(self) -> Dict[int, torch.Tensor]:
        """Compute combined activations from all active concepts."""
        combined = {}
        
        for concept, strength in self._active_concepts.values():
            activations = concept.get_activations(
                layers=self.layers,
                coefficient=strength,
                device=self.device,
                dtype=self.dtype,
            )
            
            for layer, activation in activations.items():
                if layer in combined:
                    combined[layer] = combined[layer] + activation
                else:
                    combined[layer] = activation
        
        return combined
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        do_sample: bool = True,
        temperature: float = 0.7,
        **generation_kwargs,
    ) -> str:
        """
        Generate text with active concept control.
        
        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            **generation_kwargs: Additional generation arguments.
        
        Returns:
            Generated text string.
        """
        activations = self._compute_activations() if self._active_concepts else None
        
        outputs = self.pipeline(
            prompt,
            activations=activations,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **generation_kwargs,
        )
        
        return outputs[0]['generated_text']
    
    def generate_batch(
        self,
        prompts: List[str],
        **generation_kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts."""
        activations = self._compute_activations() if self._active_concepts else None
        
        outputs = self.pipeline(
            prompts,
            activations=activations,
            **generation_kwargs,
        )
        
        return [o['generated_text'] for o in outputs]
    
    @property
    def active_concepts(self) -> Dict[str, float]:
        """Get dict of active concept names and strengths."""
        return {name: strength for name, (_, strength) in self._active_concepts.items()}


# =============================================================================
# Pre-configured Extractors
# =============================================================================

class HonestyExtractor(ConceptExtractor):
    """
    Pre-configured extractor for honesty/truthfulness detection.
    
    Example:
        >>> extractor = HonestyExtractor(model, tokenizer)
        >>> concept = extractor.extract_default()
        >>> is_truthful = extractor.is_honest("The Earth is round.")
    """
    
    DEFAULT_TEMPLATE = "Pretend you're {type} person making statements about the world."
    
    def __init__(
        self,
        model,
        tokenizer,
        user_tag: str = "",
        assistant_tag: str = "",
        **kwargs,
    ):
        super().__init__(model, tokenizer, concept_name="honesty", **kwargs)
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
    
    def extract_from_statements(
        self,
        statements: List[str],
        template: Optional[str] = None,
        **kwargs,
    ) -> ConceptDirection:
        """
        Extract honesty direction from a list of factual statements.
        
        Args:
            statements: List of factual statements.
            template: Template string with {type} placeholder. Default: standard template.
            **kwargs: Additional extraction arguments.
        """
        template = template or self.DEFAULT_TEMPLATE
        
        positive_prompts = []
        negative_prompts = []
        
        for stmt in statements:
            prefix = f"{self.user_tag} {template.format(type='an honest')} {self.assistant_tag} "
            positive_prompts.append(prefix + stmt)
            
            prefix = f"{self.user_tag} {template.format(type='an untruthful')} {self.assistant_tag} "
            negative_prompts.append(prefix + stmt)
        
        return self.extract_from_pairs(positive_prompts, negative_prompts, **kwargs)
    
    def is_honest(self, text: str, threshold: float = 0.0, **kwargs) -> bool:
        """Check if text appears honest/truthful."""
        return self.is_concept_present(text, threshold=threshold, **kwargs)
    
    def get_honesty_score(self, text: str, **kwargs) -> float:
        """Get average honesty score for text."""
        return self.get_average_score(text, **kwargs)


class EmotionExtractor(ConceptExtractor):
    """
    Pre-configured extractor for emotion detection and control.
    
    Supported emotions: happiness, sadness, anger, fear, disgust, surprise
    """
    
    EMOTIONS = {
        "happiness": (["joyful", "happy", "cheerful"], ["dejected", "unhappy", "sad"]),
        "sadness": (["sad", "depressed", "miserable"], ["cheerful", "happy", "optimistic"]),
        "anger": (["angry", "furious", "irritated"], ["calm", "peaceful", "pleased"]),
        "fear": (["fearful", "scared", "frightened"], ["fearless", "bold", "brave"]),
        "disgust": (["disgusted", "revolted", "sickened"], ["pleased", "delighted", "satisfied"]),
        "surprise": (["surprised", "shocked", "astonished"], ["unimpressed", "bored", "indifferent"]),
    }
    
    def __init__(
        self,
        model,
        tokenizer,
        emotion: str,
        user_tag: str = "",
        assistant_tag: str = "",
        **kwargs,
    ):
        if emotion not in self.EMOTIONS:
            raise ValueError(f"emotion must be one of {list(self.EMOTIONS.keys())}")
        
        super().__init__(model, tokenizer, concept_name=emotion, **kwargs)
        self.emotion = emotion
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.positive_adj, self.negative_adj = self.EMOTIONS[emotion]
    
    def extract_from_scenarios(
        self,
        scenarios: List[str],
        template: str = "Act as if you are extremely {adj}.",
        **kwargs,
    ) -> ConceptDirection:
        """
        Extract emotion direction from scenarios.
        
        Args:
            scenarios: List of scenario beginnings.
            template: Template with {adj} placeholder for emotion adjective.
            **kwargs: Additional extraction arguments.
        """
        positive_prompts = []
        negative_prompts = []
        
        for scenario in scenarios:
            pos_adj = np.random.choice(self.positive_adj)
            neg_adj = np.random.choice(self.negative_adj)
            
            pos_prefix = f"{self.user_tag} {template.format(adj=pos_adj)} {self.assistant_tag} "
            neg_prefix = f"{self.user_tag} {template.format(adj=neg_adj)} {self.assistant_tag} "
            
            positive_prompts.append(pos_prefix + scenario)
            negative_prompts.append(neg_prefix + scenario)
        
        return self.extract_from_pairs(positive_prompts, negative_prompts, **kwargs)
