 < 2:
        return None
    try:
        dt = pd.to_datetime(s)
        if pd.isna(dt):
            return None
        return dt
    except Exception:
        return None

def fix_timestamps(df):
    out_df = df.copy()
    
    act_order = ['Register Claim', 'Verify Documents', 'Evaluate Claim', 'Approve Claim', 'Close Claim']
    act_to_idx = {act: i for i, act in enumerate(act_order)}
    
    # 1. Compute median durations for consecutive activities
    deltas_sec = [[] for _ in range(4)]
    
    # Store parsed raw timestamps per case
    case_data = []
    
    for case_id, group in df.groupby('CaseID', sort=False):
        raw_vals = [None] * 5
        row_indices = [None] * 5
        for idx, row in group.iterrows():
            act = row['Activity']
            if act in act_to_idx:
                a_idx = act_to_idx[act]
                raw_vals[a_idx] = row['Timestamp']
                row_indices[a_idx] = idx
        
        parsed_ts = [parse_valid_ts(v) for v in raw_vals]
        case_data.append((row_indices, parsed_ts))
        
        for k in range(4):
            t1, t2 = parsed_ts[k], parsed_ts[k+1]
            if t1 is not None and t2 is not None and t2 > t1:
                deltas_sec[k].append((t2 - t1).total_seconds())
                
    median_deltas_sec = []
    median_deltas = []
    for k in range(4):
        if deltas_sec[k]:
            med = float(np.median(deltas_sec[k]))
        else:
            med = 3600.0  # fallback if no valid pairs
        median_deltas_sec.append(med)
        median_deltas.append(pd.Timedelta(seconds=med))
        
    # 2. Repair timestamps for each case
    repaired_series = pd.Series(index=df.index, dtype='datetime64[ns]')
    
    for row_indices, parsed_ts in case_data:
        V = [i for i, t in enumerate(parsed_ts) if t is not None]
        
        best_S = ()
        best_len = -1
        best_dev = float('inf')
        
        for r in range(len(V), 0, -1):
            for S