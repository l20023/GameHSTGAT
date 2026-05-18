# Can Graph Neural Networks Break the Social Learning Barrier?

**Authors:** Leo-Minh Kustermann, Maxime Skipetrov  
**Affiliation:** Tsinghua University / Game Theory, China  

---

## Abstract
Huang, Strack, and Tamuz (2024) proved a fundamental mathematical upper bound (HST bound) on social learning speed: regardless of network size or structure, rational agents cannot learn faster than a rate determined solely by their private signal quality. This proposal investigates whether Graph Attention Networks (GATs) can achieve or exceed this bound when tested empirically. By mapping repeated social interactions to recurrent message passing, we propose the first computational test using an anytime-predictive Generalist GNN to determine whether the HST bound is an information-theoretic limit or an artifact of rational equilibrium constraints.

---

## 1. Introduction & Problem Statement

Social learning examines how individuals in a network aggregate information from neighbors to learn an unknown state. Recently, Huang, Strack, and Tamuz (2024) proved a striking theoretical result: regardless of network size or topology, rational agents cannot learn faster than a rate capped by a constant that depends only on private signal informativeness. In their binary-signal calibration with 90% accurate private signals, this upper bound implies that even very large networks cannot learn faster than roughly a ten-agent public-signal benchmark, so most generated information is lost in equilibrium.

This proposal bridges economic theory and machine learning. Graph Neural Networks (GNNs), specifically Graph Attention Networks (GATs), perform structurally similar message passing but optimize weights via gradient descent rather than strict game-theoretic equilibrium reasoning. 

> **Core Research Question:** Can a single trained, recurrent GAT operating as a generalist agent achieve or exceed the theoretical HST learning bound when evaluated dynamically across standard graph topologies? If so, this demonstrates that the learning barrier is not information-theoretically fundamental, but rather a constraint imposed by rational agency.

---

## 2. Background & Related Work

### 2.1 The HST Model
HST model a network $G = (V, E)$ with $n$ agents. In each period $t$, agent $i$ receives a private signal $s_{i,t} \in \{a, b\}$ about a binary state $\theta \in \{A, B\}$ with quality $q = P(s_{i,t} = a \mid \theta = A) \in (0.5, 1)$. Agents observe neighbors' past actions and choose their own to maximize utility. The exponential decay of the error probability $\varepsilon(T) \sim e^{-\beta T}$ yields the learning rate $\beta$, bounded by:

$$\beta \leq M,\qquad M = 2\sup_s |\ell(s)|$$

For the binary symmetric signal model used in this proposal,
$$\ell(s)\in\left\{\pm\log\frac{q}{1-q}\right\}\quad\Rightarrow\quad \beta_{\max}(q)=M=2\log\frac{q}{1-q}.$$

This mathematical proof holds universally across all graphs, but lacks empirical verification or algorithmic baselines.

### 2.2 Graph Attention Networks (GATs)
GATs compute node representations by aggregating neighbor features using learnable attention coefficients $\alpha_{ij}$:

$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)} h_j^{(l)}\right)$$

This architecture matches the information-exchange structure of social learning, where layers correspond to time steps. No prior work has utilized GNNs to benchmark these theoretical economic bounds.

---

## 3. Proposed Methodology

### 3.1 Graph Generation
To evaluate performance across distinct structural paradigms within our tight experimental scope, we restrict our analysis to two foundational frameworks implemented via NetworkX:

*   **Complete Graph (`nx.complete_graph`):** A fully connected baseline where all $n$ agents observe each other simultaneously. This graph represents the physical upper bound of instantaneous, global information propagation.
*   **Watts-Strogatz Network (`nx.connected_watts_strogatz_graph`):** A model that allows us to interpolate between localized and connected topologies by varying the rewiring probability $p$:
    *   **Localized Ring ($p = 0$):** A regular lattice where agents only communicate with their $k=2$ immediate neighbors, restricting information to incremental, sequential local flows.
    *   **Small-World ($p = 0.1$):** A lattice where 10% of edges are randomly rewired, introducing structural shortcuts that model realistic social networks by bridging distant nodes.

By comparing the Generalist RGAT on the Complete Graph against the $p=0$ and $p=0.1$ variants of the Watts-Strogatz model, we can systematically isolate whether global visibility or local shortcuts are required to break the HST bound. All graphs assume 100% homophily regarding the ground truth label.

### 3.2 Architecture: Recurrent GAT (RGAT)
We propose a Recurrent GAT to process sequential signals over $T$ periods:

$$h_i^{(t)} = \text{GRU}\left(\left[s_{i,t}; \sum_{j \in \mathcal{N}(i)} \alpha_{ij} h_j^{(t-1)}\right], h_i^{(t-1)}\right)$$

where private signals are concatenated with aggregated neighbor messages. At each intermediate step $t$, node predictions are generated via a shared MLP head: 

$$\hat{\theta}_{i}^{(t)} = \text{MLP}(h_i^{(t)})$$

### 3.3 Training Paradigm: Shared Sequential Loss
To evaluate whether a flexible, single-model setup can adaptively learn over time like human agents, we train a single **Generalist Model**. Instead of optimizing only for the final horizon, the model is trained across all time steps simultaneously. For a maximum horizon $T_{\max} = 50$, we employ a shared sequential cross-entropy loss over all nodes $V$ and all time steps $t$:

$$\mathcal{L}_{\text{total}} = \frac{1}{T_{\max} \cdot |V|} \sum_{t=1}^{T_{\max}} \sum_{i \in V} \mathcal{L}_{\text{CE}}\left(\hat{\theta}_{i}^{(t)}, \theta\right)$$

This optimization setup forces the internal representations of the RGAT to remain useful and predictive at any given point in time, enabling a single model to act as a generalist across the entire learning horizon.

### 3.4 Evaluation
During inference, the single trained Generalist model is evaluated at every step $t \in \{1, \dots, T_{\max}\}$. We will compute the empirical error rate $\varepsilon(t)$ over 10,000 independent test trials. By tracking $\varepsilon(t)$ sequentially from a single forward pass, we fit the decay rate using non-linear least squares (`scipy.optimize.curve_fit`):

$$\varepsilon(t) = \alpha e^{-\beta_{\text{GAT}} t} + \varepsilon_{\infty}$$

The resulting empirical learning rate $\beta_{\text{GAT}}$ will be directly compared against the analytical economic bound $\beta_{\max}(q)=2\log\frac{q}{1-q}$ for the binary symmetric case.

---

## 4. Experimental Setup & Expected Outcomes

### Table 1: Experimental Hyperparameters (Proof of Concept)

| Parameter | Value(s) |
| :--- | :--- |
| Network Sizes ($n$) | 10, 50, 100 |
| Signal Quality ($q$) | 0.60, 0.80 |
| Max Horizon ($T_{\max}$) | 50 |
| Training / Test Episodes | 50,000 / 5,000 |
| Hidden Dimension | 32 |
| Attention Heads | 2 |
| Optimizer | Adam ($\eta = 0.001$) |

### Expected Empirical Regimes

We anticipate three distinct empirical regimes for our Generalist RGAT, each carrying a fundamentally different implication for social learning theory:

*   **Regime 1: Support for the Information-Theoretic Limit ($\beta_{\text{GAT}} \leq \beta_{\text{HST}}$):**  
    If the GAT cannot exceed the bound, it provides strong empirical support that the HST bound represents a hard physical limit of information propagation, suggesting that rational equilibrium reasoning—despite its inefficiencies—is asymptotically rate-optimal.
    
*   **Regime 2: Empirical Counter-Evidence via Existence Proof ($\beta_{\text{GAT}} > \beta_{\text{HST}}$):**  
    If the GAT systematically outperforms the bound, it serves as an empirical existence proof that the learning barrier is not an absolute information-theoretic ceiling, but rather a pathological constraint imposed by the assumptions of game-theoretic rationality.
    
*   **Regime 3: Boundary Conditions of the Economic Framework:**  
    If $\beta_{\text{GAT}}$ scales with network size $n$ or varies across topologies, these empirical insights would suggest that the theoretical invariance claims of the HST model rely heavily on fragile equilibrium constraints rather than structural independence.

This work provides the first empirical connection between deep learning on graphs and formal social learning bounds, offering concrete insights into how artificial networks circumvent strategic communication bottlenecks.