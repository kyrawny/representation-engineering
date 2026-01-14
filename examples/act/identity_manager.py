"""
Identity Manager - ACT Identity and Modifier Database

Manages identities and modifiers from ACT dictionaries:
- Load identities from MTurkInteract_Identities.csv
- Load modifiers from MTurkInteract_Modifiers.csv
- Apply modifiers to identities
- Re-identification: find closest matching identity to a transient EPA
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .act_core import EPA


@dataclass
class Identity:
    """An ACT identity with name and EPA profile."""
    name: str
    epa: EPA
    
    def __repr__(self) -> str:
        return f"Identity('{self.name}', {self.epa})"


@dataclass
class Modifier:
    """An ACT modifier (trait/status) with name and EPA profile."""
    name: str
    epa: EPA
    
    def __repr__(self) -> str:
        return f"Modifier('{self.name}', {self.epa})"


class IdentityDatabase:
    """
    Database of ACT identities loaded from CSV.
    
    Provides lookup by name and search functionality.
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Load identities from CSV file.
        
        Args:
            csv_path: Path to identities CSV (default: data/act/MTurkInteract_Identities.csv)
        """
        if csv_path is None:
            csv_path = Path(__file__).parent.parent.parent / "data" / "act" / "MTurkInteract_Identities.csv"
        
        self.df = pd.read_csv(csv_path)
        self._build_lookup()
    
    def _build_lookup(self):
        """Build identity lookup dictionary."""
        self.identities: Dict[str, Identity] = {}
        for _, row in self.df.iterrows():
            name = row['term']
            epa = EPA(e=row['E'], p=row['P'], a=row['A'])
            self.identities[name.lower()] = Identity(name=name, epa=epa)
    
    def get(self, name: str) -> Optional[Identity]:
        """
        Get identity by name (case-insensitive).
        
        Args:
            name: Identity name (e.g., "doctor", "friend")
            
        Returns:
            Identity if found, None otherwise
        """
        return self.identities.get(name.lower())
    
    def get_epa(self, name: str) -> Optional[EPA]:
        """Get EPA for identity by name."""
        identity = self.get(name)
        return identity.epa if identity else None
    
    def search(self, query: str, limit: int = 10) -> List[Identity]:
        """
        Search identities by partial name match.
        
        Args:
            query: Search query
            limit: Max results to return
            
        Returns:
            List of matching identities
        """
        query_lower = query.lower()
        matches = [
            identity for name, identity in self.identities.items()
            if query_lower in name
        ]
        return matches[:limit]
    
    def find_closest(
        self,
        target_epa: EPA,
        limit: int = 5,
        exclude: Optional[List[str]] = None
    ) -> List[Tuple[Identity, float]]:
        """
        Find identities closest to a target EPA (re-identification).
        
        Uses Euclidean distance in EPA space.
        
        Args:
            target_epa: Target EPA to match
            limit: Max results to return
            exclude: Identity names to exclude
            
        Returns:
            List of (Identity, distance) tuples, sorted by distance
        """
        exclude_set = set(name.lower() for name in (exclude or []))
        target = target_epa.to_array()
        
        distances = []
        for name, identity in self.identities.items():
            if name in exclude_set:
                continue
            dist = np.linalg.norm(identity.epa.to_array() - target)
            distances.append((identity, float(dist)))
        
        distances.sort(key=lambda x: x[1])
        return distances[:limit]
    
    def re_identify(self, transient_epa: EPA, exclude: Optional[List[str]] = None) -> Identity:
        """
        Re-identification: find the identity closest to a transient EPA.
        
        Per Heise's ACT, when transient impressions deviate significantly from
        fundamental sentiments, actors may "re-identify" - relabeling the 
        identity to match current impressions.
        
        Args:
            transient_epa: Current transient EPA
            exclude: Identity names to exclude from consideration
            
        Returns:
            Closest matching identity
        """
        closest = self.find_closest(transient_epa, limit=1, exclude=exclude)
        if closest:
            return closest[0][0]
        raise ValueError("No identities available for re-identification")
    
    def list_all(self) -> List[str]:
        """List all identity names."""
        return [identity.name for identity in self.identities.values()]
    
    def __len__(self) -> int:
        return len(self.identities)


class ModifierDatabase:
    """
    Database of ACT modifiers (traits/statuses) loaded from CSV.
    
    Modifiers can be applied to identities to create modified identities
    (e.g., "angry doctor", "young student").
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Load modifiers from CSV file.
        
        Args:
            csv_path: Path to modifiers CSV (default: data/act/MTurkInteract_Modifiers.csv)
        """
        if csv_path is None:
            csv_path = Path(__file__).parent.parent.parent / "data" / "act" / "MTurkInteract_Modifiers.csv"
        
        self.df = pd.read_csv(csv_path)
        self._build_lookup()
    
    def _build_lookup(self):
        """Build modifier lookup dictionary."""
        self.modifiers: Dict[str, Modifier] = {}
        for _, row in self.df.iterrows():
            name = row['term']
            epa = EPA(e=row['E'], p=row['P'], a=row['A'])
            self.modifiers[name.lower()] = Modifier(name=name, epa=epa)
    
    def get(self, name: str) -> Optional[Modifier]:
        """
        Get modifier by name (case-insensitive).
        
        Args:
            name: Modifier name (e.g., "angry", "young")
            
        Returns:
            Modifier if found, None otherwise
        """
        return self.modifiers.get(name.lower())
    
    def get_epa(self, name: str) -> Optional[EPA]:
        """Get EPA for modifier by name."""
        modifier = self.get(name)
        return modifier.epa if modifier else None
    
    def search(self, query: str, limit: int = 10) -> List[Modifier]:
        """Search modifiers by partial name match."""
        query_lower = query.lower()
        matches = [
            modifier for name, modifier in self.modifiers.items()
            if query_lower in name
        ]
        return matches[:limit]
    
    def list_all(self) -> List[str]:
        """List all modifier names."""
        return [modifier.name for modifier in self.modifiers.values()]
    
    def __len__(self) -> int:
        return len(self.modifiers)


def apply_modifier(
    identity_epa: EPA,
    modifier_epa: EPA,
    modifier_weight: float = 0.5
) -> EPA:
    """
    Apply a modifier to an identity EPA.
    
    Per Heise's ACT, modifier combination typically uses weighted averaging.
    The exact combination rules can vary; this implements a common approach.
    
    Modified_EPA = (1 - weight) * Identity_EPA + weight * Modifier_EPA
    
    Args:
        identity_epa: Base identity EPA
        modifier_epa: Modifier EPA to apply
        modifier_weight: Weight for modifier (0-1, default 0.5)
        
    Returns:
        Combined EPA
    """
    id_arr = identity_epa.to_array()
    mod_arr = modifier_epa.to_array()
    combined = (1 - modifier_weight) * id_arr + modifier_weight * mod_arr
    return EPA.from_array(combined)


def apply_modifiers(
    identity_epa: EPA,
    modifier_epas: List[EPA],
    weights: Optional[List[float]] = None
) -> EPA:
    """
    Apply multiple modifiers to an identity EPA.
    
    Args:
        identity_epa: Base identity EPA
        modifier_epas: List of modifier EPAs
        weights: Optional weights for each modifier (default: equal weights)
        
    Returns:
        Combined EPA
    """
    if not modifier_epas:
        return identity_epa
    
    if weights is None:
        # Equal weights for all modifiers
        n = len(modifier_epas)
        weights = [1.0 / (n + 1)] * n  # Leave room for identity
    
    # Start with identity contribution
    identity_weight = 1.0 - sum(weights)
    result = identity_weight * identity_epa.to_array()
    
    # Add modifier contributions
    for mod_epa, weight in zip(modifier_epas, weights):
        result += weight * mod_epa.to_array()
    
    return EPA.from_array(result)


@dataclass
class ModifiedIdentity:
    """An identity with applied modifiers."""
    base_identity: Identity
    modifiers: List[Modifier] = field(default_factory=list)
    combined_epa: Optional[EPA] = None
    
    def __post_init__(self):
        if self.combined_epa is None:
            self._compute_combined()
    
    def _compute_combined(self):
        """Compute combined EPA from base identity and modifiers."""
        if not self.modifiers:
            self.combined_epa = self.base_identity.epa
        else:
            modifier_epas = [m.epa for m in self.modifiers]
            self.combined_epa = apply_modifiers(self.base_identity.epa, modifier_epas)
    
    @property
    def epa(self) -> EPA:
        """Get the combined EPA."""
        return self.combined_epa
    
    @property
    def name(self) -> str:
        """Get display name (e.g., 'angry doctor')."""
        if not self.modifiers:
            return self.base_identity.name
        mod_names = ' '.join(m.name for m in self.modifiers)
        return f"{mod_names} {self.base_identity.name}"
    
    def __repr__(self) -> str:
        return f"ModifiedIdentity('{self.name}', {self.combined_epa})"


# Default database instances (lazy-loaded)
_default_identity_db: Optional[IdentityDatabase] = None
_default_modifier_db: Optional[ModifierDatabase] = None


def get_identity_database() -> IdentityDatabase:
    """Get or create default identity database instance."""
    global _default_identity_db
    if _default_identity_db is None:
        _default_identity_db = IdentityDatabase()
    return _default_identity_db


def get_modifier_database() -> ModifierDatabase:
    """Get or create default modifier database instance."""
    global _default_modifier_db
    if _default_modifier_db is None:
        _default_modifier_db = ModifierDatabase()
    return _default_modifier_db


def create_identity(
    identity_name: str,
    modifier_names: Optional[List[str]] = None,
    identity_db: Optional[IdentityDatabase] = None,
    modifier_db: Optional[ModifierDatabase] = None
) -> ModifiedIdentity:
    """
    Convenience function to create a (possibly modified) identity.
    
    Args:
        identity_name: Base identity name (e.g., "doctor")
        modifier_names: Optional modifier names (e.g., ["angry", "young"])
        identity_db: Identity database (uses default if None)
        modifier_db: Modifier database (uses default if None)
        
    Returns:
        ModifiedIdentity instance
        
    Raises:
        ValueError: If identity or modifier not found
    """
    identity_db = identity_db or get_identity_database()
    modifier_db = modifier_db or get_modifier_database()
    
    base = identity_db.get(identity_name)
    if base is None:
        raise ValueError(f"Identity not found: {identity_name}")
    
    modifiers = []
    for mod_name in (modifier_names or []):
        mod = modifier_db.get(mod_name)
        if mod is None:
            raise ValueError(f"Modifier not found: {mod_name}")
        modifiers.append(mod)
    
    return ModifiedIdentity(base_identity=base, modifiers=modifiers)
