IDENTITY_CSV = r"MTurkInteract_Identities.csv"
COEF_CSV = r"2010impressionformation.csv"

def impression_formation_matrix(A: Dict[str, float], B: Dict[str, float], O: Dict[str, float],
                               coef_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate impression formation using coefficient matrix from Heise's Expressive Order."""
    input_vector = np.array([A['e'], A['p'], A['a'], B['e'], B['p'], B['a'], O['e'], O['p'], O['a']])
    
    def coef_term_value(coef_name):
        """Calculate the value of a coefficient term based on binary encoding."""
        term = 1.0
        for i, ch in enumerate(coef_name[1:]):  # Skip initial 'Z'
            if ch == '1':
                term *= input_vector[i]
        return term
    
    result = {}
    for col in coef_df.columns[1:]:  # Skip coef_name column
        val = 0.0
        for _, row in coef_df.iterrows():
            term_val = coef_term_value(row['coef_name'])
            val += term_val * row[col]
        result[col] = val
    
    return result

def calculate_deflection(fundamental_epa: np.ndarray, transient_epa: np.ndarray) -> float:
    """Calculate deflection as squared Euclidean distance between fundamental and transient EPA."""
    return np.sum((fundamental_epa - transient_epa) ** 2)

def find_optimal_behavior(actor_fund: Dict[str, float], object_fund: Dict[str, float],
                         actor_trans: Dict[str, float], object_trans: Dict[str, float],
                         coef_df: pd.DataFrame,
                         bounds: List[Tuple[float, float]] = [(-4.3, 4.3)] * 3) -> Dict[str, float]:
    """Find optimal behavior EPA values that minimize total system deflection."""
    
    def objective(b_vec):
        """Objective function to minimize: total deflection after the behavior."""
        B = {'e': b_vec[0], 'p': b_vec[1], 'a': b_vec[2]}
        post = impression_formation_matrix(actor_trans, B, object_trans, coef_df)
        
        # Extract post-event EPAs
        actor_aft = np.array([post[f'postA{dim.upper()}'] for dim in ['e', 'p', 'a']])
        behavior_aft = np.array([post[f'postB{dim.upper()}'] for dim in ['e', 'p', 'a']])
        object_aft = np.array([post[f'postO{dim.upper()}'] for dim in ['e', 'p', 'a']])
        
        # Calculate deflections from fundamentals
        actor_deflection = calculate_deflection(
            np.array([actor_fund['e'], actor_fund['p'], actor_fund['a']]), actor_aft
        )
        behavior_deflection = calculate_deflection(np.array([B['e'], B['p'], B['a']]), behavior_aft)
        object_deflection = calculate_deflection(
            np.array([object_fund['e'], object_fund['p'], object_fund['a']]), object_aft
        )
        
        return actor_deflection + behavior_deflection + object_deflection
    
    # Numerical optimization to find optimal behavior
    result = minimize(objective, x0=np.zeros(3), bounds=bounds, method='L-BFGS-B')
    
    return {'e': result.x[0], 'p': result.x[1], 'a': result.x[2]}

print("Core ACT functions defined!")