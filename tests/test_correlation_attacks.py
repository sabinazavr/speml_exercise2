# test_correlation_attacks.py

import pandas as pd
from correlation_attacks.src.attacks import column_wise_attack
from scheme._universal import Universal

def test_column_attack_removes_marks():
    df = pd.DataFrame({'X': [0,1,0,1]*25})
    u = Universal(gamma=10, fingerprint_bit_length=8, xi=2)
    df_fp = u.insertion(dataset=df, recipient_id=1, secret_key=123)
    # simulate attacker knowing uniform joint
    uniform = df_fp.groupby(['X','X']).size().unstack(fill_value=0)/100
    df_attacked = column_wise_attack(df_fp, {('X','X'): uniform}, threshold=0.0)
    # after attack, detection should fail more often
    detected = u.detection(dataset=df_attacked, secret_key=123)
    assert detected != 1  # at least sometimes wrong
