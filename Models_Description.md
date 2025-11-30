
# Multirotor UAV Multi-Fault Detection Using Lightweight Deep Learning with Single and Double Fault Magnitudes

This repository implements two different lightweight deep learning architectures, **CLFDNet** and **AELMFNet**, for fault detection and diagnosis in multirotor UAVs. Both models operate on time-series sensor data and are designed to be efficient enough for potential onboard deployment.

- **CLFDNet** – hybrid CNN–LSTM with attention for **discrete fault-magnitude classification** (single and double faults).
- **AELMFNet** – LSTM autoencoder with attention for **reconstruction-based anomaly detection** and **continuous fault severity estimation**.

---

## Key Features

- Multi-branch **1D CNN** for multi-scale temporal feature extraction.
- **LSTM** branches to model sequential UAV dynamics.
- **Attention mechanisms** to focus on the most informative time steps.
- Support for **single** and **double** motor faults and multiple fault magnitudes.
- Comparison of **six different loss functions** for AELMFNet.
- Lightweight architectures suitable for resource-limited platforms.

---

## 1. Problem Description

Multirotor UAVs are susceptible to actuator and sensor faults that can compromise performance or result in loss of control. Early detection of these faults and estimation of their severity is critical for safe and robust operation.

This repository focuses on:

1. **Classifying** discretized fault magnitudes (e.g., 5%, 10%, 15%, …) for single and double motor faults using CLFDNet.
2. **Estimating** fault severity continuously by modeling normal behavior and measuring reconstruction error using AELMFNet.

The models operate on **time windows of UAV telemetry**, including attitude, angular rates, accelerations, motor commands, and related signals.

---

## 2. CLFDNet Architecture (Classification Network)

**CLFDNet** is a dual-branch architecture combining a **multi-branch 1D CNN** with an **LSTM + attention** branch. It is trained end-to-end for **multiclass fault-magnitude classification**.

### 2.1. Input Data

Each input window contains a short segment of UAV telemetry, typically covering around one second of flight. Features can include, for example:

- Angular rates: `p, q, r`
- Linear accelerations: `ax, ay, az`
- Euler angles: `phi, theta, psi`
- Motor outputs: `m1, m2, m3, m4`
- Time or index feature

All features are **z-score normalized**. Windows are centered around fault onset, allowing for the capture of both pre-fault and post-fault behavior.

Two views of the same window are used as inputs:

- `X_cnn` – shaped for the CNN branch.
- `X_lstm` – shaped for the LSTM branch (same sequence, different input tensor).

---

### 2.2. CNN Branch: Multi-Scale Temporal Features

The CNN branch is designed to capture **local temporal patterns** and fault signatures at multiple scales.

**Core ideas:**

- Use **six parallel 1D convolution paths** to process the same input sequence.
- Each path has different filter sizes and numbers of filters to capture:
  - Short-term transients (small kernels).
  - Longer-term trends (larger kernels).
  - Multiple levels of feature richness (different filter counts).

**Typical structure:**

- Input: sequence of length `T` with 1 or more channels.
- Six Conv1D blocks in parallel, for example:
  - Paths with kernel size 3 and 32 filters.
  - Paths with kernel size 5 and 48 filters.
  - One path with kernel size 3 and 64 filters.
- Each Conv1D block:
  - Uses ReLU activation.
  - It is followed by **Global Average Pooling** to reduce the temporal dimension and retain the most informative activations.
- The outputs of all six paths are:
  - Concatenated into one **multi-scale feature vector**.
  - Passed through:
    - Dense layer (32 units, ReLU).
    - Dropout layer (e.g., 0.05) for regularization.
    - Dense layer (32 units, ReLU).

This produces a compact vector (CNN feature vector) that summarizes local temporal structures in the signal.

---

### 2.3. LSTM Branch: Sequential Dynamics

The LSTM branch focuses on **global temporal dynamics** and long-range dependencies.

**Structure:**

- Input: same time-series window as the CNN branch (same length and features).
- One LSTM layer with 32 units:
  - Configured with `return_sequences=True` to keep the full sequence of hidden states.
- Output: a sequence of hidden states, one per time step, representing how the system evolves over the window.

This branch is responsible for modeling the **progression** of the UAV state over time, including gradual degradations or transitions.

---

### 2.4. Attention Mechanism

To better combine CNN and LSTM information, CLFDNet uses an **additive attention** mechanism:

- The CNN feature vector acts as a **context/query**.
- The LSTM sequence of hidden states acts as the **value/key sequence**.
- For each time step in the LSTM output, an attention score is computed based on:
  - How compatible the LSTM state is with the CNN context vector.
- These scores are normalized into attention weights (using softmax).
- A **context vector** is then computed as a weighted sum of all LSTM hidden states.

Effectively:

- The model learns **which time steps** in the sequence are most relevant given the features extracted by the CNN.
- This improves the ability to focus on segments like:
  - Fault onset.
  - Rapid changes in motor behavior.
  - Significant deviations from normal flight.

The attention-enhanced representation is then reduced to a fixed-length vector (e.g., via pooling or direct combination).

---

### 2.5. Feature Fusion and Classification

The final classification layer combines information from both branches.

1. Take:
   - The compact CNN feature vector.
   - The attention-augmented LSTM feature vector.
2. Concatenate them into a **fusion vector**.
3. Pass the fusion vector through:
   - Dense layer (64 units, ReLU).
   - Output Dense layer with `num_classes` units and **softmax** activation.

The output is a probability distribution over **discretized fault-magnitude classes**.

---

### 2.6. Fault Magnitude Discretization

Fault magnitudes are discretized into **fixed intervals** (e.g., 5% steps). For example:

- [2.5%, 7.5%) → class representing 5%
- [7.5%, 12.5%) → class representing 10%
- and so on.

Intermediate values are assigned to the nearest interval. CLFDNet is trained to predict these class labels for:

- **Single faults** (one motor affected).
- **Double faults** (two motors affected simultaneously).

---

## 3. AELMFNet Architecture (Autoencoder Network)

**AELMFNet** is an LSTM-based **sequence-to-sequence autoencoder** with soft attention. Instead of directly classifying faults, it learns to **reconstruct normal (or faulted) sequences** and uses reconstruction error as a measure of anomaly or fault severity.

This is useful when:

- You want a **continuous estimate** of fault magnitude.
- You want to detect **unseen fault conditions** or subtle degradations.

---

### 3.1. Overall Structure

AELMFNet consists of:

1. **Encoder** – stacked LSTM layers compress the input sequence into a low-dimensional latent representation.
2. **Attention module** – allows the decoder to focus on specific parts of the encoded sequence.
3. **Decoder** – stacked LSTM layers reconstruct the original time-series from the latent representation (and attention context).

---

### 3.2. Encoder: Latent Representation

- Input: multivariate time-series window with the same features used in CLFDNet.
- Multiple LSTM layers are stacked with **decreasing number of units** (e.g., 100 → 75 → 50 → 25).
- The final encoder LSTM outputs a **latent sequence**, and a low-dimensional latent vector (e.g., 3 dimensions) is derived to summarize the entire window.

The latent dimension (for example, 3) is selected to balance:

- Reconstruction accuracy.
- Complexity and memory footprint for onboard/edge deployment.

This latent vector is intended to capture the **most important temporal dynamics** of the input signal.

---

### 3.3. Soft Attention in the Autoencoder

To avoid the limitations of a fixed-size bottleneck, AELMFNet includes a **soft attention** mechanism:

- The encoder produces a sequence of hidden states.
- The decoder, at each time step, computes attention scores over all encoder states.
- These scores are converted into attention weights.
- A **context vector** is computed as a weighted sum of encoder states and combined with the decoder’s hidden state.

This allows the decoder to:

- Dynamically “look back” at different parts of the input sequence.
- Focus more on time regions that are most informative for reconstruction.
- Better handle long sequences and long-range dependencies.

---

### 3.4. Decoder: Sequence Reconstruction

- The latent representation is **repeated** across the time axis to match the window length.
- It is processed, together with attention context, by stacked LSTM layers with **increasing units** (e.g., 25 → 50 → 75 → 100).
- A final **time-distributed Dense layer** outputs a reconstructed feature vector at each time step.

The decoder’s objective is to reconstruct the input sequence as accurately as possible.

---

### 3.5. Loss Functions and Fault Severity

AELMFNet is trained under **six different loss functions**, all implemented under the same architecture and training conditions:

- Binary Cross-Entropy (BCE)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber loss
- Smooth L1 loss
- Hinge or hinge-like loss

The reconstruction loss can be interpreted as:

- An **anomaly score** – higher loss usually indicates that the input is less similar to the training data distribution (e.g., a more severe or unusual fault).
- A **proxy for fault severity** – reconstruction error tends to increase with higher fault magnitudes.

By comparing these different loss functions, we can understand how each one affects:

- Sensitivity to small deviations.
- Robustness to noise.
- Overall performance for fault detection and severity estimation.

---

## 4. Relationship Between CLFDNet and AELMFNet

The two models are intended to be **complementary**:

- **CLFDNet**:
  - Provides **fast, discrete** decisions about fault classes and magnitudes.
  - Suitable when predefined fault levels are available (e.g., 5%, 10%, 15%).
- **AELMFNet**:
  - Provides **continuous, reconstruction-based** measures of fault severity.
  - Useful for detecting subtle anomalies, unseen states, or gradations within and between discrete fault levels.

Together, they form a flexible framework for multi-fault detection and diagnosis in multi-rotor UAVs.

---
