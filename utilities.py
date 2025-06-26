import numpy as np
from scipy.linalg import expm

def van_loan_discretise_Q(A, Qc, Ts, G=None):
    n = A.shape[0]
    if G is None:
        G = np.eye(n)

    # Build Van-Loan block matrix
    M = np.block([
        [ -A,           G @ Qc @ G.T ],
        [ np.zeros_like(A),  A.T     ]
    ]) * Ts

    # Matrix exponential
    Phi = expm(M)

    # Extract blocks
    Ad  = Phi[n:, n:].T          # (should equal expm(A*Ts))
    Qd  = Ad @ Phi[:n, n:]       # lower-left block times Ad

    return Qd