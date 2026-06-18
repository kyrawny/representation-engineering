"""
ACT Pipeline — End-to-End Orchestrator.

Ties together EPA reading, ACT deflection minimisation, and EPA steering
into a single class that can process user messages and generate
affect-appropriate responses.

Typical usage::

    pipe = ACTPipeline(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        agent_identity=EPA(e=1.5, p=1.0, a=0.5),
        user_identity=EPA(e=1.0, p=0.5, a=0.3),
    )
    pipe.load_model()
    pipe.load_directions("epa_directions.pkl")
    pipe.setup_reader("epa_reading_tuning_v2_results.json")
    pipe.setup_steerer(base_coeff=2.0)

    response = pipe.process_message("What the hell is wrong with you?")
    print(response)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .act_core import (
    EPA,
    ACTCoefficients,
    get_default_coefficients,
    get_response_epa_for_deflection_minimization,
)
from .direction_extraction import load_directions
from .epa_reader import EPAReader
from .epa_steerer import EPASteerer
from .prompt_formatting import format_llama3_prompt


class ACTPipeline:
    """
    End-to-end pipeline: read user EPA → ACT compute → steer response.

    Encapsulates model loading, direction loading, EPA reading, ACT
    deflection minimisation, and steered generation.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        agent_identity: Optional[EPA] = None,
        user_identity: Optional[EPA] = None,
        coefficients: Optional[ACTCoefficients] = None,
        system_prompt: str = "You are a human person engaging in a conversation with another person. Keep your responses concise.",
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            agent_identity: Agent's fundamental EPA identity.
            user_identity: User's fundamental EPA identity.
            coefficients: ACT impression-formation coefficients.
                If None, loads the default 2010 coefficients.
            system_prompt: System prompt used when formatting generation
                prompts.
        """
        self.model_name = model_name
        self.agent_identity = agent_identity or EPA(e=1.0, p=1.0, a=0.5)
        self.user_identity = user_identity or EPA(e=1.0, p=0.5, a=0.3)
        self.coefficients = coefficients or get_default_coefficients()
        self.system_prompt = system_prompt

        # Components (populated by setup methods)
        self.model = None
        self.tokenizer = None
        self.rep_readers: Optional[Dict[str, Any]] = None
        self.hidden_layers: Optional[List[int]] = None
        self.reader: Optional[EPAReader] = None
        self.steerer: Optional[EPASteerer] = None
        self.rep_reading_pipeline = None

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def load_model(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_directions(self, path: str = "epa_directions.pkl") -> None:
        """
        Load pre-extracted EPA direction vectors.

        Args:
            path: Path to the pickle file from ``save_directions()``.
        """
        saved = load_directions(path)
        self.rep_readers = saved["rep_readers"]
        self.hidden_layers = saved["hidden_layers"]

    def setup_reader(
        self,
        results_path: str = "epa_reading_tuning_v2_results.json",
        method: str = "ElasticNet",
    ) -> None:
        """
        Configure the calibrated EPA reader from tuning results.

        Args:
            results_path: Path to reading tuning results JSON.
            method: Layer selection method name.
        """
        assert self.rep_readers is not None, "Call load_directions() first."

        from transformers import pipeline as hf_pipeline
        from repe import repe_pipeline_registry

        repe_pipeline_registry()

        self.rep_reading_pipeline = hf_pipeline(
            "rep-reading", model=self.model, tokenizer=self.tokenizer,
        )
        self.reader = EPAReader.from_tuning_results(
            results_path, self.rep_readers, method=method,
        )

    def setup_steerer(self, base_coeff: float = 2.0) -> None:
        """
        Configure the EPA steerer from the reader config.

        Args:
            base_coeff: Default steering coefficient per layer.
        """
        assert self.reader is not None, "Call setup_reader() first."
        assert self.model is not None, "Call load_model() first."

        self.steerer = EPASteerer.from_reader(
            reader=self.reader,
            rep_readers=self.rep_readers,
            model=self.model,
            tokenizer=self.tokenizer,
            base_coeff=base_coeff,
        )

    # -----------------------------------------------------------------
    # Core pipeline
    # -----------------------------------------------------------------

    def read_user_epa(self, user_message: str) -> Dict[str, float]:
        """
        Read the EPA of a user message.

        Args:
            user_message: Raw user text.

        Returns:
            Dict with ``'evaluation'``, ``'potency'``, ``'activity'`` values.
        """
        assert self.reader is not None, "Call setup_reader() first."
        return self.reader.read_epa(self.rep_reading_pipeline, user_message)

    def compute_target_epa(self, user_behavior_epa: Dict[str, float]) -> EPA:
        """
        Use ACT to compute the optimal EPA for the agent's response.

        Takes the user's behaviour EPA, runs impression formation and
        deflection minimisation, and returns the target behaviour EPA
        for the agent's reply.

        Args:
            user_behavior_epa: Dict with ``'evaluation'``, ``'potency'``,
                ``'activity'`` keys from reading the user's message.

        Returns:
            Optimal ``EPA`` for the agent's response.
        """
        behavior_epa = EPA(
            e=user_behavior_epa["evaluation"],
            p=user_behavior_epa["potency"],
            a=user_behavior_epa["activity"],
        )
        return get_response_epa_for_deflection_minimization(
            agent_identity=self.agent_identity,
            user_identity=self.user_identity,
            user_behavior_epa=behavior_epa,
            coefficients=self.coefficients,
        )

    def generate_response(
        self,
        user_message: str,
        target_epa: Optional[Union[EPA, Dict[str, float]]] = None,
        max_new_tokens: int = 128,
        **generation_kwargs,
    ) -> str:
        """
        Generate a steered response to a user message.

        If *target_epa* is not provided, it is computed automatically
        via ``compute_target_epa()``.

        Args:
            user_message: Raw user text.
            target_epa: Target EPA for steering (computed if omitted).
            max_new_tokens: Maximum tokens.
            **generation_kwargs: Extra arguments.

        Returns:
            Generated response text.
        """
        assert self.steerer is not None, "Call setup_steerer() first."

        if target_epa is None:
            user_epa = self.read_user_epa(user_message)
            target_epa = self.compute_target_epa(user_epa)

        if isinstance(target_epa, EPA):
            target_dict = {
                "evaluation": target_epa.e,
                "potency": target_epa.p,
                "activity": target_epa.a,
            }
        else:
            target_dict = target_epa

        prompt = format_llama3_prompt(self.system_prompt, user_message)

        return self.steerer.generate(
            prompt=prompt,
            target_epa=target_dict,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )

    def process_message(
        self,
        user_message: str,
        max_new_tokens: int = 128,
        **generation_kwargs,
    ) -> str:
        """
        Full pipeline: read user EPA → ACT compute → steer generation.

        This is the main entry point for the complete pipeline.

        Args:
            user_message: Raw user text.
            max_new_tokens: Maximum tokens.
            **generation_kwargs: Extra arguments.

        Returns:
            Generated response text, steered toward the ACT-optimal EPA.
        """
        return self.generate_response(
            user_message,
            target_epa=None,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
