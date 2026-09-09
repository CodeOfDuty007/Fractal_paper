#This module implements Multifractal Analysis on a 1D discrete time series (token surprise scores). Its primary objective is to extract high-dimensional geometric features—Self-Similarity and Long-Range Dependence to create a statistical "fingerprint" of the input text.
import numpy as np
import json
#importing library for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.stats import linregress

scale = [8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320]
#Epsilon is NOT a fixed constant handed down by any single reference --
#three sources give three different values: andsteing/gagle's decompiled
#bundle.js defaults to 1e-2 (exposed as a UI slider, so not fixed there
#either); the GAGLE_Dataset.ipynb demo uses 0.1, noting explicitly this is
#"larger than in the paper" because its concatenation method differs; and
#the ORIGINAL paper's own colab (google-research/fractals_language/
#fractal_patterns_in_language.ipynb, arXiv:2402.01825) uses 5e-2 as an
#"illustration only" default on 896-1024 token documents with scales up to
#300 -- and explicitly states its own defaults differ from what the actual
#paper experiments used. Epsilon has to be calibrated to the corpus's own
#document length / scale-range ratio, not copied from any one source.
#Calibrated empirically here via a sweep against real GAGLE corpus-level
#fits (see reports/ for the sweep result and justification).
EPSILON = 1e-2
#Per-scale bootstrap draw count. NOT the same thing as the "bootstrap-10" in
#Extension1_PhaseWise_Execution_Plan.md, which refers to outer CI-resampling
#repeats of the whole estimation, not this. Verified against synthetic fBm
#with known Hurst exponent: at n_boot=10 on a pooled/long series, bias
#ranged -0.25 to -0.31 with R^2 0.13-0.65 (pure sampling noise -- each
#p_epsilon(tau) is estimated from as few as 10 draws). Sweeping n_boot
#10->100->1000->5000 on the same series collapsed bias to -0.005..-0.06
#with R^2 0.89-1.0. 2000 is comfortably in the region where bias is
#sampling-noise-dominated no longer, without the cost of 5000.
N_BOOT = 2000

def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data
def generate_trajectory(RawScores):
    scores=np.array(RawScores)
    mu=np.mean(scores)
#calc standard deviation
    sigma=np.std(scores)
    #z-score before cumsumming: verified byte-for-byte against the reference
    #andsteing/gagle Space's `computeXx` ((x-mean)/std then cumsum). An
    #earlier session mistakenly "fixed" this by removing the /sigma, on the
    #theory that z-scoring decoupled a fixed epsilon from the trajectory's
    #real scale -- but the reference relies on exactly this normalization to
    #make a single fixed epsilon meaningful across documents of different
    #raw score scale. That earlier change is reverted here.
    increment=(scores-mu)/sigma
    #below line of code creates the random walk using the cumsum of increment
    trajectory=np.cumsum(increment)
    return trajectory,increment

def calc_S(x, scale_param=None, epsilon=EPSILON, n_boot=N_BOOT, return_std_err=False):
    """x: a single trajectory (1D array-like), OR a list of trajectories from
    multiple documents to pool into one corpus-level estimate. The paper is
    explicit that S/H describe a stochastic process and are meant to be
    estimated from a corpus of many documents, not a single ~400-token
    realization. This list-of-trajectories form matches the ORIGINAL
    paper's own reference code (google-research/fractals_language/
    fractal_patterns_in_language.ipynb, arXiv:2402.01825 -- the source the
    GAGLE dataset's own notebook says it "copied/modified" its version
    from): get_self_sim_exponent there takes a 2D (docs x tokens) array and
    draws random offsets WITHIN each document row, never crossing document
    boundaries. GAGLE_Dataset.ipynb's own demo instead concatenates all
    documents into one flat 1D array before sampling, which lets a random
    (offset, offset+tau) pair straddle a document boundary and measure a
    meaningless displacement -- a shortcut deviation from the original
    method, not something to replicate.
    """
    current_scale = scale_param if scale_param is not None else scale
    trajectories = x if (isinstance(x, list) and len(x) > 0 and hasattr(x[0], "__len__")) else [x]
    trajectories = [np.asarray(t) for t in trajectories]

    flat = np.concatenate(trajectories) if len(trajectories) > 1 else trajectories[0]
    if len(flat) < 2:
        return (float("nan"), 0.0, float("nan")) if return_std_err else (float("nan"), 0.0)
    #second normalization, distinct from generate_trajectory's raw-score z-score:
    #divide the trajectory itself by its own (pooled) stddev, matching the
    #reference's `l = a(s); r = r.map(row => row.map(v => v/l))`.
    traj_std = np.std(flat)
    if traj_std > 0:
        trajectories = [t / traj_std for t in trajectories]

    #Bootstrap p_epsilon(tau) = P(|X_{t+tau}-X_t| < epsilon), matching the
    #reference exactly: n_boot RANDOM start offsets per scale per trajectory,
    #not the exhaustive sliding-window enumeration a single ~400-token
    #document doesn't have enough independent samples to support at large tau
    #(observed directly: 3-of-272 non-overlapping-equivalent hits at tau=128).
    #Vectorized (draw all n_boot offsets for a (trajectory, tau) pair in one
    #np.random.randint call, then fancy-index all displacements at once)
    #rather than a Python-level loop of n_boot individual randint() calls --
    #same sampling distribution, ~identical statistical behavior, but P1
    #calls this ~1.8M times (vs. O(10^2-10^3) calls in P0/fBm validation),
    #where the per-call Python-loop overhead alone was projected at ~10
    #hours just for this step.
    p0 = []
    for tau in current_scale:
        disps = []
        for traj in trajectories:
            n = len(traj)
            if n <= tau:
                continue
            k = np.random.randint(0, n - tau, size=n_boot)
            disps.append(traj[k + tau] - traj[k])
        #disps is now a list of per-trajectory (n_boot,) arrays -- concatenate
        #explicitly (rather than relying on np.abs(list-of-arrays) auto-
        #stacking) so the mean is unambiguously over every trajectory's draws
        #pooled together, matching the old flat-list-of-scalars behavior.
        all_disps = np.concatenate(disps) if disps else np.array([])
        p0.append(np.mean(np.abs(all_disps) < epsilon) if len(all_disps) else 0.0)

    #log-log regression, filtering non-positive scale/prob pairs (matching the
    #reference's power-law fit helper, which applies the same filter).
    scale_arr = np.asarray(current_scale, dtype=float)
    p0_arr = np.asarray(p0, dtype=float)
    mask = (scale_arr > 0) & (p0_arr > 0)
    if mask.sum() < 2:
        return (float("nan"), 0.0, float("nan")) if return_std_err else (float("nan"), 0.0)
    slope, intercept, r_value, p_value, std_err = linregress(np.log(scale_arr[mask]), np.log(p0_arr[mask]))
    s = -slope
    #Coefficient of Determination) tells you how well your data actually fits the mathematical model you are trying to use.
    #R^2 represents the proportion of the variance in the dependent variable
    #t tells you how much the text actually behaves like a fractal.
    #std_err here is linregress's standard error of the SLOPE estimate --
    #a legitimate, free-of-charge OLS uncertainty on Hat{S} itself (S = -slope),
    #used by build_p1_feature_store.py as a per-document CI half-width
    #(1.96*std_err for a 95% CI) instead of an expensive repeated-bootstrap
    #outer loop, since the inner n_boot resampling already dominates cost.
    if return_std_err:
        return s, r_value**2, std_err
    return s, r_value**2



def calc_H(increment, scale_param=None, return_std_err=False):
    """scale_param: explicit scale list to sweep, defaulting to the module-level
    `scale`. Needed for windowed sub-document H (P1 Family A): a fixed-size
    window (e.g. 200 tokens) can't support the full 11-value scale up to 320
    -- large scales silently drop out (len//n == 0 gives zero chunks for that
    n), and building log_n from a front-truncated slice of the GLOBAL scale
    list (the previous behavior) then pairs the surviving R/S values against
    the WRONG scales whenever a scale is skipped for any reason other than
    being the largest ones. This only ever worked by luck on 320-400 token
    docs, where the dropped scales happened to be exactly the largest, at the
    end of the list. Explicit scale_param + building log_n from the actually-
    used scales (not a prefix slice) fixes both a real bug and a latent one.
    """
    current_scale = scale_param if scale_param is not None else scale
    used_scales = []
    avg_rs_values = []
    for n in current_scale:
        chunks=len(increment)//n
        rs_ratio=[]
        for i in range(chunks):
            #divide an article into smaller chunks
            chk=increment[i*n:(i+1)*n]
            z = np.cumsum(chk - np.mean(chk))
            R = np.max(z) - np.min(z)
            S = np.std(chk)
            if S > 0:
                rs_ratio.append(R / S)
        if len(rs_ratio) > 0:
            avg_rs_values.append(np.mean(rs_ratio))
            used_scales.append(n)

    if len(avg_rs_values) < 2:
        return (float("nan"), 0.0, float("nan")) if return_std_err else (float("nan"), 0.0)
    log_n = np.log(used_scales)
    log_rs = np.log(avg_rs_values)
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_rs)
    H = slope
    #std_err: OLS standard error of the slope (=H) from the same regression --
    #see calc_S's return_std_err for why this is used as a free CI half-width
    #instead of a repeated-bootstrap outer loop.
    if return_std_err:
        return H, r_value**2, std_err
    return H, r_value**2

def calc_cosine(x,y):
    #this step converts the list into a [1,n] ordered matrix
    x = x.reshape(1,-1)
    y = y.reshape(1,-1)
    #the below function automatically normalizes the vectors.
    return cosine_similarity(x,y)[0][0]

#now we analyse semantic drift, embeddings is the dimensions of a particular item 
# window size is set to find and avererage value to compare text
def analyse_semanticDrift(embeddings, window_size=50):
    num_block= len(embeddings)//window_size
    block=[]
    for i in range(num_block):
        # this process is called Temporal Pooling or Semantic Chunking
        #we slice the embeddings into blocks of size window_size and calculate the mean embedding for each block. 
        #This mean embedding represents the "average" semantic content of that block of text.
        block_slice=embeddings[i*window_size:(i+1)*window_size]
        block.append(np.mean(block_slice, axis=0))
        
    if len(block)<2:
        print("Not enough blocks to analyze semantic drift.")
        return None, None
        
    simm=[]
    for i in range(len(block)-1):
        sim=calc_cosine(block[i],block[i+1])
        simm.append(sim)
    return np.mean(simm), np.var(simm)

def get_allMetric(scores, embedding=None):
    trajectory,increment= generate_trajectory(scores)
    H, H_r2 = calc_H(increment)
    #calc_S here runs on a SINGLE ~400-token document -- verified against
    #synthetic fBm with known Hurst exponent to be severely biased in this
    #regime (bias -0.26 to -0.47, R^2 0.10-0.38) even with well-calibrated
    #n_boot. calc_S only recovers the ground-truth exponent accurately when
    #applied to a long/pooled trajectory (many documents' raw scores
    #concatenated, then z-scored+cumsummed ONCE), matching how the paper and
    #the andsteing/gagle reference tool actually use it (corpus-level, not
    #per-document). Holder_Exp below is kept for backward compatibility with
    #existing consumers but is flagged explicitly rather than silently
    #presented as trustworthy at the per-document grain this function runs at.
    S, S_r2 = calc_S(trajectory)
    results = {
        "Hurst_Exp": float(H),
        "Hurst_R2": float(H_r2),
        "Holder_Exp": float(S),
        "Holder_R2": float(S_r2),
        "Holder_PerDoc_Unreliable": True
    }
    if embedding is not None:
        sem_mean, sem_var = analyse_semanticDrift(embedding)
        results["Semantic_Consistency"] = float(sem_mean)
        results["Semantic_Volatility"] = float(sem_var)
    return results
if __name__ == "__main__":
    import os
    from collections import Counter

    IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "new_fractal_scores.jsonl")
    OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "new_analysis_results_v2.jsonl")

    processed_count = 0
    ecount = 0
    error_types = Counter()
    # Open both files simultaneously to stream data without clogging RAM
    with open(IN_PATH, 'r') as fin, open(OUT_PATH, 'w') as fout:
        for line in fin:
            try:
                entry = json.loads(line)
                sample_id = entry.get('id', 'unknown_id')
                scores = entry.get('scores', [])
                if not scores or len(scores) < 320:
                    ecount += 1
                    continue
                # Calculate the math
                metrics = get_allMetric(scores, embedding=None)
                # Package the results along with the metadata needed for plotting
                entry_result = {
                    "id": sample_id,
                    "metrics": metrics,
                    "Label": entry.get("model_used", "Human"),
                    "temp": entry.get("temp", None),
                    "prompt_type": entry.get("prompt_type", entry.get("prompt type", "continue")),
                    "domain": entry.get("domain", "unknown")
                }
                # Write immediately to disk
                fout.write(json.dumps(entry_result) + "\n")
                processed_count += 1
                if processed_count % 5000 == 0:
                    print(f"Calculated metrics for {processed_count} articles...")
            except Exception as e:
                ecount += 1
                error_types[type(e).__name__] += 1
    print(f"\nSuccessfully processed {processed_count} entries into {OUT_PATH}.")
    print(f"Skipped {ecount} entries. Error breakdown: {dict(error_types)}")

