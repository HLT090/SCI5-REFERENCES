# ASGMamba: Adaptive Spectral Gating Mamba for Multivariate Time Series Forecasting

Qianyang Li<sup>1</sup>, Xingjun Zhang<sup>1\*</sup>, Shaoxun Wang<sup>1</sup>, Jia Wei<sup>2</sup>, Yueqi Xing<sup>3</sup>

<sup>1\*</sup>School of Computer Science and Technology, Xi’an Jiaotong University, Xi’an 710049, China.

<sup>2</sup>Department of Computer Science and Technology, Tsinghua University, Beijing 100084, China.

<sup>3</sup>Department of Research and Development, Shandong New Beiyang Information Technology Co., Ltd, Weihai 264200, China.

\*Corresponding author(s). E-mail(s): xjzhang@xjtu.edu.cn; Contributing authors: liqianyang@stu.xjtu.edu.cn; shaoxunwang@stu.xjtu.edu.cn; weijia4473@mail.tsinghua.edu.cn; xingyueqi@newbeiyang.com;

## Abstract

Long-term multivariate time series forecasting (LTSF) plays a crucial role in various high-performance computing applications, including real-time energy grid management and large-scale trafic flow simulation. However, existing solutions face a dilemma: Transformer-based models sufer from quadratic complexity, limiting their scalability on long sequences, while linear State Space Models (SSMs) often struggle to distinguish valuable signals from high-frequency noise, leading to wasted state capacity. To bridge this gap, we propose ASGMamba, an eficient forecasting framework designed for resource-constrained supercomputing environments. ASGMamba integrates a lightweight Adaptive Spectral Gating (ASG) mechanism that dynamically filters noise based on local spectral energy, enabling the Mamba backbone to focus its state evolution on robust temporal dynamics. Furthermore, we introduce a hierarchical multi-scale architecture with variable-specific Node Embeddings to capture diverse physical characteristics. Extensive experiments on nine benchmarks demonstrate that ASGMamba achieves state-of-the-art accuracy. While keeping strictly

O(L)

complexity we significantly reduce the memory usage on long-horizon tasks, thus establishing ASGMamba as a scalable solution for high-throughput forecasting in resource limited environments.The code is available at https://github.com hit636/ASGMamba

Keywords: Time Series Forecasting , State Space Models , Mamba , Spectral Analysis

## 1 Introduction

Long-term multivariate time series forecasting (LTSF) is essential in applications such as decision-making in complex systems [1], ranging from high-frequency financial markets [2, 3] to real-time energy grid management [4, 5]. With the exponential growth of sensor data, deploying accurate forecasting models on high-performance computing (HPC) infrastructures has become a critical challenge. In these operational environments, a model is defined not only by its statistical accuracy but also by its computational eficiency [6, 7]. Real-world data streams typically exhibit entangled multi-scale dynamics, where critical long-term trends are obscured by high-frequency stochastic noise [8, 9]. Consequently, the central challenge in LTSF is to distinguish structurally informative signals from noise and capture long-range dependencies without incurring prohibitive memory or latency costs.

In recent years, Transformer-based architectures have dominated LTSF [10–12]. While leveraging global self-attention enables efective modeling of historical dependencies, these models sufer from a quadratic computational complexity $\mathcal { O } ( L ^ { 2 } )$ with respect to the sequence length L. Beyond theoretical complexity, the heavy memory footprint of attention maps often saturates GPU bandwidth for ultra-long sequences, creating a bottleneck for high-throughput applications. Sparse attention variants, such as Informer [13], attempt to reduce this to O(L log L) via heuristic sampling, but often at the cost of discarding fine-grained signal information. However, Structured State Space Models (SSMs) [14, 15] ofer a more eficient alternative, enabling linear scaling for long-range temporal dependencies. By framing sequence modeling as a discretized continuous-time system [16], Mamba achieves strict linear scaling O(L) through a hardware-aware parallel scan, ofering a rigorous foundation for processing massive temporal contexts.

Although SSMs ofer eficiency benefits, applying standard Mamba architectures to noisy time series demonstrates a surprising robustness-eficiency conflict. The standard selective scan mechanism processes input tokens sequentially in the time domain. However, distinguishing valid high-frequency signals from stochastic noise purely in the time domain requires the model to approximate complex filtering operations, which expends significant state capacity. Depleted of latent state information, the limited latent state of the SSM is often saturated by high-frequency noise, leaving little capacity left to model the underlying long-term trend [17]. We therefore contend this is a state eficiency problem: compelling a linear recurrent system to approximate global spectral filtering from scratch is computationally suboptimal compared to supplying an explicit, lightweight spectral prior.

Furthermore, to preserve linear complexity, many architectures decouple multivariate dependencies via a Channel-Independent (CI) protocol [18]. Although eficient, this strategy inherently discards semantic context, as the model learns a shared representation for all variates regardless of their physical nature. Treating heterogeneous signals like voltage and temperature with identical dynamics inevitably degrades performance when their underlying spectral properties difer significantly.

To resolve these conflicts, we propose ASGMamba (Adaptive Spectral Gating Mamba), a linear-complexity framework that conditions state evolution on spectral energy distribution. Many existing methods either forecast directly in the frequency domain or apply global Fourier transforms [19, 20] with O(L log L) cost, breaking support for streaming prediction. Diferent from these methods, ASGMamba follows the local frequency analysis to condition adaptive gating. From a system viewpoint, our Adaptive Spectral Gating (ASG) is a data-dependent frequency-selective filter that suppresses noise-dominated components at the input stage with local patch energy information, thereby forcing Mamba backbone to reserve its state capacity for robust dynamics. Importantly, by applying FFT on fixed-size patches, we strictly maintain O(L) complexity. Furthermore, we employ learnable Node Embeddings to recover variable-specific semantics in the shared backbone without resorting to quadratic channel mixing.

The main contributions of this work are summarized as follows:

• Spectral-Conditioned State Evolution Framework: We propose ASGMamba, which bridges the gap between spectral analysis and linear recurrence. By conditioning the SSM input on local spectral energy density, we prevent high-frequency noise from contaminating the latent state, significantly enhancing the efective capacity of the model for trend modeling.

• Input-Dependent Linear Filtering Mechanism: We introduce the Adaptive Spectral Gating (ASG) module. Functioning as a learnable filter, it dynamically modulates input fidelity based on frequency properties. This acts as a computationeficient prior that efectively separates signal from noise within a strict O(L) budget.

• Hierarchical Multi-Scale System Design: We implement a multi-branch architecture augmented with learnable Node Embeddings. This design enables the model to capture multi-granularity temporal patterns and distinct physical semantics of variables simultaneously, ensuring scalability for high-dimensional multivariate data.

• Eficiency-Accuracy Trade-of: Extensive evaluations on nine real-world benchmarks demonstrate that ASGMamba achieves competitive or superior performance compared to state-of-the-art Transformer and SSM baselines. Notably, it exhibits significantly reduced memory footprint and inference latency in long-sequence scenarios, validating its suitability for resource-constrained supercomputing environments.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 shows the preliminaries. Section 4 details the architecture of ASGMamba. Section 5 presents our experimental setup and results. Finally, Section 6 concludes the paper.

## 2 Related Work

This section reviews literature closely related to our work, organized into three research streams: deep learning methods for long-term time series forecasting (LTSF), state space models (SSMs) for eficient sequence modeling, and frequency-domain analysis in deep neural networks.

## 2.1 Deep Learning Methods for Time Series Forecasting

Most data-driven forecasting approaches prior to deep learning were statistical models such as ARIMA and Prophet [21], which adopt linear dynamics assumption and are hard to model the nonlinear patterns frequently presented in large-scale sensor data. When deep learning emerges, RNNs, LSTMs and GRUs become widely used deep models for sequential modeling due to their recurrent property. However, their recurrent nature makes them hard to be parallelized and thus becomes ineficient to model ultra-long sequences, especially in high sampling frequency scenarios [8]. Transformer [22] leverages self-attention to model global dependency in sequence without recurrent property. Follow-up methods like LogTrans [10] and Informer [13] also reduce the quadratic complexity of attention by leveraging sparse attention mechanism and further apply decomposition strategy to model trend and seasonal components independently [19, 23]. Patch-based methods like PatchTST [10] aggregate local temporal segments into tokens to enhance local pattern modeling and reduce sequence length. Proposed as an eficient MLP-based solution, the MSTF [24] model enhances longterm time series forecasting by combining a Time Reverse and Transform block for global perception with a Dynamic Combination Reconstruction block to efectively capture multi-scale temporal dependencies. Nevertheless, these Transformer-based models still adopt attention mechanism whose computational cost and memory cost both increase superlinearly with sequence length, which motivates us to explore more scalable architectures for long-horizon forecasting.

## 2.2 State Space Models and Mamba

State Space Models (SSMs) have recently re-emerged as an eficient alternative to attention-based architectures, aiming to combine parallelizable training with fast autoregressive inference. The Structured State Space model (S4) [25] demonstrated that imposing structured parameterizations on state transitions enables SSMs to model long-range dependencies with linear complexity. Building upon S4, Mamba [14] introduced a selective scanning mechanism that allows model parameters to be inputdependent, achieving strong performance in large-scale sequence modeling tasks while preserving linear scaling.

In the context of time series forecasting, recent studies have explored adapting SSMs to multivariate and long-horizon settings. For example, S4ND [26] extended S4 to multidimensional signals, and TimeMachine [27] applied Mamba-style architectures to forecasting tasks. However, these approaches primarily operate in the time domain and do not explicitly account for the spectral characteristics of sensor data. As a result, high-frequency noise and informative periodic components are processed within a shared state space, potentially limiting robustness in highly volatile environments.

## 2.3 Frequency-Domain Analysis in Deep Learning

Frequency-domain analysis has been applied in many fields to describe periodicity and filter out noise from time series. In deep learning, FEDformer [19] applied Fourier transforms in combination with attention mechanisms to model global dependencies more eficiently. TimesNet [28] represented time series in a two-dimensional space according to the dominant periods, and FITS [29] applied forecasting in the complex frequency domain. Recently, MSFMoE [30] applied multi-scale frequency filtering in combination with a mixture-of-experts architecture to improve long-term forecasting performance.

Although these frequency-based methods have achieved promising performance, most of them apply spectral transformations in combination with global feature extraction or static filtering. That is, they transfer the entire sequence using fixed mappings and do not modulate the information flow via finer time grains in an adaptive way. Therefore, we attempt to explore how the spectral information can be applied as a structural gating signal in an SSM framework.

## 3 Preliminaries

In this section, we formulate the forecasting problem and provide the theoretical background of the Structured State Space Model (SSM), which serves as the backbone of our eficient framework.

## 3.1 Problem Formulation

The task of Multivariate Time Series (MTS) forecasting aims to predict future trajectories based on historical observations. Let $\mathcal { D } = \{ ( \mathbf { X } ^ { ( i ) } , \mathbf { Y } ^ { ( i ) } ) \} _ { i = 1 } ^ { \mathcal { N } }$ denote a dataset of time series samples. For a single instance i, $\mathbf { X } ^ { ( i ) } = \left\{ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { L } \right\} \in \mathbb { R } ^ { L \times M }$ represents the historical look-back sequence, where L is the observation window length and M is the number of variates (channels). The objective is to predict the future sequence $\mathbf { Y } ^ { ( i ) } = \{ \mathbf { x } _ { L + 1 } , \dots , \mathbf { x } _ { L + T } \} \in \mathbb { R } ^ { T \times M }$ over a horizon T.

Formally, we seek to learn a mapping function $\mathcal { F } _ { \boldsymbol { \theta } } : \mathbb { R } ^ { L \times M }  \mathbb { R } ^ { T \times M }$ , parameterized by θ, that minimizes the prediction discrepancy on unseen data. To leverage parallel computing hardware, the model processes inputs in mini-batches. Consequently, the input to the algorithm is represented as a tensor $\mathbf { X } _ { i n } \in \mathbb { R } ^ { B \times L \times M }$ , where B denotes the batch size. Furthermore, under the Channel-Independent (CI) strategy, this batch is reshaped to $\mathbb { \ R } ^ { ( B \cdot M ) \times L \times 1 }$ to share backbone parameters across all variates eficiently.

## 3.2 Eficient Modeling via State Space Models

The core eficiency of ASGMamba stems from the Structured State Space Model (SSM) [25]. SSMs map a 1D input stimulation $x ( t ) \in \mathbb { R }$ to an output $y ( t ) \in \mathbb { R }$ through a latent state $h ( t ) \in \mathbb { R } ^ { N }$ . The system is modeled as a linear Time-Invariant (LTI) continuous system:

$$
\dot {h} (t) = \mathbf {A} h (t) + \mathbf {B} x (t), \quad y (t) = \mathbf {C} h (t),\tag{1}
$$

where $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ is the evolution matrix, and B, C are projection parameters. To handle discrete time series, the system is discretized using a time scale parameter ∆, transforming Eq. (1) into a recurrence relation:

$$
h _ {t} = \bar {\mathbf {A}} h _ {t - 1} + \bar {\mathbf {B}} x _ {t}, y _ {t} = \mathbf {C} h _ {t}.\tag{2}
$$

This recursive form allows for fast autoregressive inference (O(1) per step). Alternatively, the system can be computed via global convolution during training, enabling parallelization similar to Transformers.

In State Space Models (SSMs), all parameters are typically fixed. In contrast, the Mamba [14] model makes the parameter set B, C, ∆ functions of $x _ { t } .$ . Hence, Mamba can decide at each time step which parameters to propagate or forget along the sequence $( \mathrm { e . g . }$ , skip connections for zeroing out noise). Despite the input-dependence, Mamba is still able to evaluate Eq. (2) eficiently via a convolution-like operation using a hardware-aware parallel scan algorithm. More importantly, due to its linear computational complexity $O ( L )$ , this architecture is significantly more eficient than the quadratic $O ( L ^ { 2 } )$ attention mechanism used in Transformers when processing long historical sequences.

## 4 Proposed Method

This section presents the architecture of ASGMnable (Adaptive Spectral Gating Mamba). Moreover, the overarching design objective may suggest that spectralconditioned state evolution should be enforced within a strictly linear computational budget. Given that standard State Space Models (SSMs) provide eficient $\mathcal O ( L )$ inference, the time-domain selective scan mechanism appears ineficient at filtering broadband noise. However, noise may lead to state saturation. To resolve this, ASG-Mamba injects a lightweight spectral prior directly into the backbone, functioning as an input-modulated filter that disentangles valid signals from noise prior to state encoding.

As illustrated in Fig. 1, the model employs a multi-level approach with adaptive patching strategies to preserve the identity of the data. To handle non-stationarity in raw sensor data, we apply Reversible Instance Normalization (RevIN) [31] at the instance level. Following normalization, we adopt a Channel-Independent (CI) strategy, reshaping the input to $\tilde { \mathcal { X } } \in \mathbb { R } ^ { ( B \cdot N ) \times L \times 1 }$ . This treats all N variates as independent sequences, reducing parameter complexity from $\mathcal { O } ( N ^ { 2 } )$ to O(1). To reduce confusion in the Channel-Independent approach, we introduce learnable node embeddings to better handle diferent variable types. The algorithmic workflow is formally detailed in Alg.1.

## 4.1 Multi-Scale Patching with Identity Preservation

Real-world systems evolve across diverse temporal granularities. To align the model’s receptive field with these physical scales, ASGMamba employs a multi-branch architecture with $K = 3$ branches, corresponding to patch sizes $P _ { k } \in \{ 8 , 1 6 , 3 2 \}$ . This design is motivated by the observation that diferent spectral bands require diferent efective temporal resolutions.

![](images/fig1.jpg)  
Fig. 1 System Architecture of ASGMamba. The framework employs a multi-scale parallel design $( P ^ { ' } = \{ 8 , 1 6 , 3 2 \} )$ ). The core innovation is the Adaptive Spectral Gating (ASG) module within the Scale Block. By computing local spectral energy density via patch-level FFT, ASG generates a data-dependent gate G. This gate acts as a frequency-selective filter, modulating the input U before it enters the Mamba encoder, thereby preventing high-frequency noise from contaminating the latent state space.

## 4.1.1 Overlapping Patching Process

Standard non-overlapping patching creates hard boundaries that can sever local dependencies. From a signal processing perspective, such discontinuities introduce high-frequency artifacts (spectral leakage) when applying FFT. To mitigate this, we adopt an overlapping strategy with stride $\begin{array} { r } { S _ { k } = \frac { \bar { P _ { k } } } { 2 } } \end{array}$ (50% overlap). This redundancy ensures that boundary information in patch i becomes central information in patch i + 1, acting as a smoothing regularizer that preserves spectral continuity for the subsequent gating module. The input sequence is unfolded into $N _ { k }$ patches, yielding a tensor $\mathbf { X } _ { p a t c h } ^ { ( k ) } \in \mathbb { R } ^ { ( B \cdot N ) \times N _ { k } \times P _ { k } }$ , which is then projected to a D-dimensional latent space $\mathbf { Z } _ { r a w } ^ { ( k ) }$

## 4.1.2 Injecting Position and Variable Identity

To compensate for the information loss caused by the CI strategy, we inject explicit semantic identifiers:

• Positional Embedding $( { \bf E } _ { p o s } )$ : A learnable tensor informs the SSM of global temporal ordering.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Forward Pass of ASGMamba
Input: Batch input  $X_{in} \in R^{B \times L \times N}$ 
Output: Forecast  $Y_{out} \in R^{B \times T \times N}$ 

1: Preprocessing:
2:  $X \leftarrow \text{RevIN}(X_{in}^{\top}) \in \mathbb{R}^{B \times V \times N}$ 
3:  $Y_{set} \leftarrow \emptyset$ 
4: for patch size  $P \in \{8, 16, 32\}$  do
5: // Patching and Context Injection
6:  $X_{p} \leftarrow \text{OverlappingPatching}(X, P)$ 
7:  $U \leftarrow \text{PatchEmb}(X_{p}) + E_{pos} + E_{node}$ 
8: // Adaptive Spectral Gating
9:  $F \leftarrow |rFFT(X_{p})|^{2}$  {Local Spectral Density}
10:  $v_{spec} \leftarrow \text{BandAgg}(F)$  {Low/Mid/High Energy}
11:  $G \leftarrow \text{Sigmoid}(\text{MLP}(v_{spec}))$ 
12: // Spectral-Conditioned SSM
13:  $U_{gated} \leftarrow U \odot G$  {Noise Suppression}
14:  $H \leftarrow Mamba(U_{gated}) + U_{gated}$ 
15: // Projection
16:  $Y^{(P)} \leftarrow \text{Head}(H)$ 
17:  $Y_{set} \leftarrow Y_{set} \cup \{Y^{(P)}\}$ 
18: end for
19: Scale Fusion &amp; Output:
20:  $Y_{fused} \leftarrow \sum_{k} \text{Softmax}(w_{scale})_{k} \cdot Y^{(k)}$ 
21:  $Y_{out} \leftarrow RevIN^{-1}(Y_{fused})^{\top}$ 
22: return  $Y_{out}$
</div>

• Node Embedding $\left( { { \bf { E } } _ { n o d e } } \right)$ : To resolve the identity crisis of CI models, we introduce a learnable matrix $\mathbf { E } _ { n o d e } \in \mathbb { R } ^ { M \times D }$ . The vector $\mathbf { e } _ { m } = \mathbf { E } _ { n o d e } [ m ]$ serves as a static semantic descriptor for variate m. By adding this embedding to the input, we condition the shared Mamba backbone to adapt its state dynamics to the specific physical properties of each variable $( \mathrm { e . g . }$ , distinct periodicities of voltage vs. load) without the quadratic cost of inter-channel attention.

The final context-aware input is formulated as:

$$
\mathbf {Z} _ {0} ^ {(k)} = \mathbf {Z} _ {r a w} ^ {(k)} + \mathbf {E} _ {p o s} + \mathbf {E} _ {n o d e}.\tag{3}
$$

## 4.2 Adaptive Spectral Gating Mamba Layer

This module constitutes the core contribution of our framework. It addresses the limitation of standard SSMs, which process all input tokens with equal weight, leading to eficient but indiscriminate modeling. We introduce a spectral filtering stage prior to the state space equation.

## 4.2.1 Adaptive Spectral Gating (ASG)

The Spectral Gating mechanism acts as a filter that adjusts based on the input data, allowing the model to focus on relevant frequencies. It assesses the spectral quality of each local patch and attenuates noise-dominated segments before they are encoded into the latent state.

## Local Spectral Transformation.

We apply a real-valued Fast Fourier Transform (rFFT) along the temporal dimension of each patch $\mathbf { X } _ { \mathrm { p a t c h } } ^ { ( k ) }$ :

$$
\mathcal {F} _ {\text { patch }} = \mathrm{rFFT} (\mathbf {X} _ {\text { patch }} ^ {(k)}) \in \mathbb {C} ^ {(B \cdot N) \times N _ {k} \times (\lfloor P _ {k} / 2 \rfloor + 1)}.\tag{4}
$$

Instead of applying global Fourier Transforms that scale at $\mathcal { O } ( L$ log L) and disrupt continuous data processing, the ASGMamba model incorporates a lightweight spectral gating mechanism, focusing on local operations to improve eficiency.

## Band-wise Energy Aggregation.

To obtain a robust spectral descriptor, we compute the spectral power density $S = | \mathcal { F } _ { \mathrm { p a t c h } } | ^ { 2 }$ and aggregate it into three coarse frequency bands. This partition is motivated by empirical observations of time series spectral concentration:

• Low Band $( [ 0 , \frac { 1 } { 3 } f _ { \mathbf { N y q } } ] )$ : Captures long-term trends and DC components.

• Mid Band $\begin{array} { r l } { ( ( \frac { 1 } { 3 } , \frac { 2 } { 3 } ] f _ { \mathbf { N y q } } ) } & { { } } \end{array}$ : Encodes dominant periodicities and seasonality.

• High Band $( ( \textstyle \frac { 2 } { 3 } , 1 ] f _ { \mathbf { N y q } } )$ : Corresponds to rapid fluctuations, often dominated by sensor noise or stochastic jitter.

The spectral descriptor $\mathbf { v } _ { \mathrm { s p e c } } \in \mathbb { R } ^ { 3 }$ is derived by summing normalized energy within these bands. This low-dimensional representation allows the gating mechanism to reason about signal composition (e.g., ”noisy” vs. ”trend-driven”) without overfitting to specific frequency bins.

## Gate Generation.

A lightweight MLP maps the spectral descriptor to a gating tensor:

$$
\mathbf {G} = \sigma (\mathbf {W} _ {g 2} \operatorname{ReLU} (\mathbf {W} _ {g 1} \mathbf {v} _ {\mathrm{spec}})),\tag{5}
$$

where $\sigma ( \cdot )$ is the Sigmoid function. The gate $\mathbf { G } \in ( 0 , 1 ) ^ { ( B \cdot M ) \times N _ { k } \times D }$ efectively acts as a confidence score: patches with high noise energy (High Band dominance) result in lower gate values $( \mathbf { G } \to 0 )$ , while trend-rich patches are preserved $( \mathbf { G }  1 )$

## 4.2.2 Spectral-Conditioned Mamba Block

We integrate the spectral gate into a residual Mamba block. Let $\mathbf { Z } _ { \mathrm { i n } }$ be the input embedding. The gated representation is computed as:

$$
\mathbf {Z} _ {\text { gated }} = \text { LayerNorm } (\mathbf {Z} _ {\text { in }}) \odot \mathbf {G}.\tag{6}
$$

This modulated signal drives the selective State Space Model (SSM):

$$
\mathbf {H} _ {t} = \mathbf {A} \mathbf {H} _ {t - 1} + \mathbf {B} \mathbf {Z} _ {\mathrm{gated}, t}, \quad \mathbf {Y} _ {t} = \mathbf {C} \mathbf {H} _ {t}.\tag{7}
$$

System Interpretation: By attenuating $\mathbf { Z } _ { \mathrm { g a t e d } }$ via $\mathbf { G } ,$ we efectively reduce the magnitude of the input projection Bx for noise-dominated tokens. This prevents the recurrent state $\mathbf { H } _ { t }$ from updating its dynamics based on spurious fluctuations, thereby conserving state capacity for valid long-term dependencies. The block concludes with a residual connection to facilitate gradient flow: $\mathbf { Z } _ { \mathrm { { o u t } } } = \mathrm { { D r o p o u t } } ( \mathrm { { M a m b a } } ( \mathbf { Z } _ { \mathrm { { g a t e d } } } ) ) + \mathbf { Z } _ { \mathrm { { i n } } }$

## 4.3 Adaptive Multi-Scale Fusion

To accommodate variables with diverse temporal dynamics $( \mathrm { e . g . }$ , slow-moving trends vs. rapid cycles), we fuse predictions from the three parallel branches $( P \in \{ 8 , 1 6 , 3 2 \} )$ ). Let $\dot { \mathbf { Y } } _ { k }$ be the forecast from branch k. We employ a learnable convex combination parameterized by $\mathbf { w } _ { s c a l e } \in \mathbb { R } ^ { 3 }$

$$
\hat {\mathbf {Y}} _ {f u s e d} = \sum_ {k = 1} ^ {3} \frac {\exp (\mathbf {w} _ {s c a l e} ^ {(k)})}{\sum_ {j = 1} ^ {3} \exp (\mathbf {w} _ {s c a l e} ^ {(j)})} \cdot \hat {\mathbf {Y}} _ {k}.\tag{8}
$$

This mechanism allows the model to dynamically prioritize the temporal resolution that best fits the data’s inherent frequency characteristics $( \mathrm { e . g . }$ , favoring larger patches for trend-dominated series).

## 4.4 Complexity and Scalability Analysis

A critical requirement for deploying forecasting models in supercomputing environments is the ability to scale eficiently with sequence length L. Here, we formally derive the computational complexity of ASGMamba.

Linear Complexity of Spectral Gating. Standard global Frequency-domain methods (e.g., FEDformer) rely on global FFT, incurring a complexity of $\mathcal { O } ( L$ log L). In contrast, ASGMamba applies $\mathrm { F F T }$ on local patches of a fixed size $P \left( \mathrm { e . g . } , P = 1 6 \right)$ Let $N \approx L / S$ be the number of patches, where S is the stride. The total complexity for the gating mechanism is:

$$
\mathcal {C} _ {\text { gate }} = N \times \mathcal {O} (P \log P) \approx \frac {L}{S} \cdot P \log P.\tag{9}
$$

Since the patch size P and stride $S$ are small constants $( P \ll L )$ , the term $\frac { P \log P } { S }$ acts as a constant multiplier. Consequently, the asymptotic complexity is strictly linear with respect to the sequence length, $\mathrm { i . e . , } O ( L )$

Implementation Details. The Adaptive Spectral Gating (ASG) module is implemented with high computational eficiency. (1) Frequency Bands: We partition the spectrum into $K _ { f r e q } = 3$ equal bands (Low, Mid, High) based on the Nyquist frequency to capture trend, periodicity, and noise, respectively. (2) Gating MLP: The spectral energy vector is processed by a 2-layer MLP with a bottleneck structure. The hidden dimension is set to $D / 4$ (reduction ratio $r = 4 )$ with ReLU activation, followed by a Sigmoid function to output valid gate weights in (0, 1). (3) Memory Eficiency: Combined with the Mamba backbone’s linear scaling $\mathcal O ( L )$ , ASGMamba avoids the quadratic memory matrix $\mathcal { O } ( L ^ { 2 } )$ of Transformers, enabling high-throughput inference on GPU clusters for ultra-long sequences $( L > 1 0 ^ { 3 } )$ .

## 5 Experiments

In this section, we systematically study ASGMamba. First, we evaluate ASGMamba on nine public datasets and demonstrate its superior forecasting accuracy compared to state-of-the-art models. Secondly, we conduct ablative analysis to illustrate the efectiveness of each component. Finally, we analyze the sensitivity of ASGMamba to some critical hyper-parameters.

## 5.1 Experimental Setup

## 5.1.1 Datasets

In order to thoroughly evaluate our method, we select nine publicly available real-world benchmarks covering various domains, sampling frequencies, and temporal patterns (Statistical information of datasets is provided in Table 1). These nine benchmarks present various forecasting challenges. Specifically, high-dimensional modeling contains Electricity and Trafic which respectively record hourly consumption from hundreds of electric meters and hourly trafic volume on thousands of roads, thus challenging the model to learn fine-grained dynamics and multi-scale seasonality on top of complex spatial-temporal correlations. For fine-grained dynamics, Weather (10-minute intervals) tests model’s robustness to nonlinear interactions between variables as well as inherent noises of sensors, and Solar-Energy records power generation of 137 photovoltaic plants, challenging models to model intense volatility from clouds coverage and strict diurnal periodicity. For addressing non-stationarity and volatility, Exchange-Rate shows structural breaks and low signal-to-noise ratio due to global economic conditions, while ETT family (ETTh1, ETTh2, ETTm1, ETTm2) contains electricity consumption measured from transformer at hourly and 15-minute granularity. ETT is more challenging due to its stronger non-stationarity and mixture of regular periodicity and irregular anomalies caused by diferent operating conditions. In total, these nine benchmarks test models on dependencies, noise, and shifts.

To evaluate the performance of ASGMamba comprehensively, we select 10 representative state-of-the-art models as baselines. These models encompass the mainstream architectural paradigms discussed in Section 2, including SSM-based, Transformer-based, and MLP/CNN-based approaches.

SSM-based Models: We compare ASGMamba with the following baselines to demonstrate its architectural benefits over standard State Space Models:

S-Mamba [32]: A straightforward adaptation of Mamba for time series. It lacks the dynamic spatial reordering feature of our proposal, generally processing variables in a fixed order or independently (Channel-Independent).

Table 1 Statistics of the benchmark datasets used in experiments.

<table><tr><td>Dataset</td><td>Variates</td><td>Timesteps</td><td>Frequency</td><td>Domain</td></tr><tr><td>Electricity</td><td>321</td><td>26,304</td><td>1-hour</td><td>Energy</td></tr><tr><td>Traffic</td><td>862</td><td>17,520</td><td>1-hour</td><td>Traffic</td></tr><tr><td>Weather</td><td>21</td><td>52,696</td><td>10-min</td><td>Weather</td></tr><tr><td>Exchange-Rate</td><td>8</td><td>7,588</td><td>Daily</td><td>Economy</td></tr><tr><td>Solar</td><td>137</td><td>7,176</td><td>1-hour</td><td>Solar</td></tr><tr><td>ETTh1 / ETTh2</td><td>7</td><td>17,420</td><td>1-hour</td><td>Energy</td></tr><tr><td>ETTm1 / ETTm2</td><td>7</td><td>69,680</td><td>15-min</td><td>Energy</td></tr></table>

Transformer-based Models: These models represent the mainstream SOTA benchmarks, using self-attention to capture long-range temporal relationships:

iTransformer [11]: It adopts an inverted structure where the whole series of a variate serves as a token. This design enables the attention module to capture multivariate correlations from a global perspective.

PatchTST [10]: A leading Channel-Independent method that divides series into patches. Although it achieves eficiency and retains local information, it does not account for inter-variable dependencies.

Crossformer [33]: It implements a two-stage attention mechanism to capture dependencies in both time and dimension (variate) axes explicitly.

FEDformer [19]: By combining seasonal-trend decomposition with Fourier Transform-based frequency attention, it eficiently models global temporal structures.

MLP- and CNN-based Models: This category includes eficient models utilizing inductive biases like decomposition and multi-scale analysis:

TimeMixer [34]: A multi-scale architecture that decomposes time series into trend and seasonality, handling them via specialized MLP pathways.

TimesNet [28]: By reshaping 1D series into 2D tensors based on multiple periods, it employs 2D convolutions to extract intra-period and inter-period features.

TiDE [35] TiDE is an MLP-based encoder-decoder model for long-term forecasting. It encodes historical data and covariates via dense layers, completely avoiding attention mechanisms. Featuring global residual connections, it achieves state-of-theart accuracy with linear computational complexity, ofering significantly faster training and inference than Transformer-based approaches.

DLinear [36]: A simple yet efective baseline that decomposes time series into trend and remainder components, processing each with a single linear layer. It challenges the necessity of complex architectures for certain forecasting tasks.

## 5.1.2 Implementation Details

Our experimental framework is rigorously designed to ensure reproducibility and to validate the eficiency advantages of the ASGMamba architecture. The implementation details are categorized into three primary dimensions: experimental environment setup, model-specific hyperparameters, and the training optimization protocol.

Experimental Setup. All experiments are implemented in PyTorch on a single NVIDIA RTX 4090 (24GB) GPU. Following standard LTSF protocols, we fix the lookback window at $L = 9 6$ and evaluate forecasting performance across four prediction horizons $H \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ on all datasets.

Model and Training Configuration. The detailed hyperparameter settings are summarized in Table 2. ASGMamba employs a multi-scale architecture with three parallel branches $( P \in \{ 8 , 1 6 , 3 2 \} )$ . Crucially, we adopt an overlapping patching strategy with a stride of $S _ { k } = P _ { k } / 2$ (50% overlap) to mitigate boundary artifacts and ensure smoother token transitions. The gating mechanism utilizes a 2-layer bottleneck MLP to map the 3-dimensional spectral energy vector into gating weights. Training is performed using the Adam optimizer. Due to the rapid convergence of the Mamba backbone, training is capped at 10 epochs. To prevent overfitting, we apply L2 weight decay, dropout, and an early stopping mechanism (patience=5). All results are averaged over five independent runs.

Table 2 Summary of Hyperparameters for ASGMamba.

<table><tr><td>Category</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="5">Model Architecture</td><td>Latent Dimension ( $D_{model}$ )</td><td>128</td></tr><tr><td>Patch Sizes (P)</td><td>{8, 16, 32}</td></tr><tr><td>Mamba State Dim ( $d_{state}$ )</td><td>16</td></tr><tr><td>Mamba Conv Kernel ( $d_{conv}$ )</td><td>4</td></tr><tr><td>Expansion Factor (E)</td><td>2</td></tr><tr><td rowspan="5">Training &amp; Opt.</td><td>Batch Size</td><td>32</td></tr><tr><td>Max Epochs</td><td>10</td></tr><tr><td>Initial Learning Rate</td><td> $10^{-3}$ </td></tr><tr><td>L2 Weight Decay</td><td> $1 \times 10^{-5}$ </td></tr><tr><td>Dropout Rate</td><td>0.1</td></tr></table>

## 5.1.3 Evaluation Metrics

We evaluate our model using two metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE). Given a prediction $\hat { Y }$ and the corresponding ground truth Y over a horizon of length H, these metrics are defined as:

$$
\mathrm{MSE} = \frac {1}{H} \sum_ {i = 1} ^ {H} (\hat {Y _ {i}} - Y _ {i}) ^ {2}\tag{10}
$$

$$
\mathrm{MAE} = \frac {1}{H} \sum_ {i = 1} ^ {H} | \hat {Y _ {i}} - Y _ {i} |\tag{11}
$$

For both metrics, lower values indicate better forecasting accuracy.

<sub>SE</sub>M<sup>A</sup> <sub>SE</sub>M<sup>AE</sup> <sub>SE</sub>M<sup>A</sup> <sub>SE</sub>M<sup>A</sup> <sub>SE</sub>M<sup>A</sup> <sub>SE</sub>M<sup>A</sup> <sub>EMAE</sub>M<sup>SEM</sup> <sub>MAE</sub>M<sup>SEM</sup> e<sup>tri</sup> <sub>EDfor</sub>m<sup>e</sup> <sub>L</sub>i<sup>nea</sup> <sub>ime</sub><sup>sNe</sup> <sub>i</sub>D<sup>e</sup> <sub>ossfor</sub>m<sup>e</sup> <sub>atchT</sub>S<sup>T</sup> <sub>ransfor</sub>m<sup>e</sup> <sub>me</sub>M<sup>ixer</sup> M<sup>amb</sup> <sub>G</sub>M<sup>amb</sup> <sub>o</sub><sup>dels</sup>

Avg0.2440.2700.2510.2760.2400.2710.2580.2780.2650.2850.2640.3200.2710.3200.2590.2870.2650.3150.3090.360 7200.3440.3390.3500.3450.3390.3410.3580.3470.3560.3490.3790.4010.3510.3860.3650.3590.3590.3450.4030.428 0.2650.2910.2740.2970.2510.2780.2780.2960.2840.3010.2730.3320.2870.3350.2800.3060.2820.3310.3390.380 <sub>3</sub><sup>6</sup> <sub>eat</sub>h<sup>er</sup> 1920.2060.2460.2140.2520.2080.2500.2210.2540.2340.2650.2090.2770.2420.2980.2190.2610.2370.2820.2760.336 960.1610.2070.1650.2100.1630.2090.1740.2120.1860.2270.1950.2710.2020.2610.1720.2200.1950.2520.2170.296

0.1470.2330.1390.2350.1530.2470.1480.2400.1900.2960.2190.3140.2370.3290.1680.2720.2100.3050.1690.273 1920.1560.2490.1590.2550.1660.2560.1620.2530.1960.3040.2310.3220.2360.3300.1840.2980.2100.3050.2010.315 Electricity3360.1740.2710.1760.2720.1850.2770.1780.2690.2170.3190.2460.3370.2490.3440.1980.3000.2230.3190.2000.304 0.2110.2950.2040.2980.2250.3170.2250.3100.2580.3520.2800.3630.2840.3730.2200.3200.2580.3500.2460.3556 <sub>2</sub><sup>0</sup>

Avg0.3510.3930.3670.4080.3910.4530.3600.4030.3670.4040.9400.7070.3700.4130.4160.4430.3540.4140.5190.429 7200.8330.6880.8670.7030.9340.7610.8470.6910.9010.7141.7671.0680.8520.6980.9640.7460.8390.6951.1950.695 0.3150.3920.3320.4180.3530.4730.3310.4170.3010.3971.2680.8830.3490.4310.3670.4480.3130.4270.4600.427 <sub>3</sub>3<sup>6</sup> <sub>xch</sub>a<sup>nge</sup> 1920.1730.2930.1820.3040.1870.3430.1770.2990.1760.2990.4700.5090.1840.3070.2260.3440.1760.3150.2710.315 0.0830.2010.0860.2070.0900.2350.0860.2060.0880.2050.2560.3670.0940.2180.1070.2340.0880.2180.1480.278 6

Avg0.2310.2650.2400.2730.2160.2800.2330.2620.2700.3070.6410.6390.3470.4170.3010.3190.3300.4010.2910.3817200.2470.2870.2600.2880.2230.2850.2490.2750.2890.3170.7690.7650.3700.4250.3380.3370.3560.4130.3570.4273360.2460.2740.2580.2880.2310.2920.2480.2730.2900.3150.7500.7350.3680.4300.3190.3300.3530.4150.2820.376 <sub>o</sub>l<sup>ar</sup>1920.2110.2570.2370.2700.2220.2830.2330.2610.2670.3100.7340.7250.3390.4160.2960.3180.3200.3980.2850.380960.2020.2340.2050.2440.1890.2590.2030.2370.2340.2860.3100.3310.3120.3990.2500.2920.2900.3780.2420.342

0.4190.2610.3820.2610.4620.2850.3950.2710.5260.3470.6440.4290.8050.4930.5930.3210.6500.3960.5870.366 1920.4280.2820.3960.2670.4730.2960.4170.2760.5220.3320.6650.4310.7560.4740.6170.3360.5980.3700.6040.373 3360.4660.2940.4170.2760.4980.2960.4330.2980.5170.3340.6740.4200.7620.4770.6290.3360.6050.3730.6210.383 7200.4820.3010.4600.3000.5060.3130.4670.3020.5520.3520.6830.4240.7190.4490.6400.3500.6450.3940.6260.382 Avg0.4780.2860.4140.2760.4840.2980.4280.2870.5290.3410.6670.4260.7600.4730.6200.3360.6250.3830.6100.3766fi<sup>c</sup>

<sub>0</sub>.<sub>4890</sub>.<sub>4980</sub>.4<sup>820</sup>.<sup>5030</sup>.<sup>4910</sup>.<sup>5440</sup>.<sup>5170</sup>.<sup>6530</sup>.<sup>6210</sup>.<sup>5940</sup>.<sup>5580</sup>.<sup>5210</sup>.<sup>5000</sup>. <sub>0</sub>.<sub>4890</sub>.<sub>4680.4</sub>5<sup>80</sup>.<sup>4580</sup>.<sup>4870</sup>.<sup>4580</sup>.<sup>5460</sup>.<sup>4960</sup>.<sup>4960</sup>.<sup>4700</sup>.<sup>5650</sup>.<sup>5150</sup>.<sup>4910</sup>.<sup>4910</sup> <sub>30</sub>.<sub>4370</sub>.<sub>4290</sub>.<sub>4210</sub>.<sub>4410</sub>.<sub>5120</sub>.<sub>4770</sub>.4<sup>290</sup>.<sup>4710</sup>.<sup>4740</sup>.<sup>5250</sup>.<sup>4920</sup>.<sup>4360</sup>.<sup>4460</sup>.<sup>4</sup> <sub>0</sub>.<sub>4050</sub>.<sub>3750</sub>.<sub>4000</sub>.<sub>3860</sub>.<sub>4050</sub>.<sub>4600</sub>.<sub>447</sub><sup>0</sup>.<sup>4230</sup>.<sup>4480</sup>.<sup>4790</sup>.<sup>4640</sup>.<sup>3840</sup>.<sup>4020</sup>.

<sub>0</sub>.<sub>4050</sub>.<sub>3810</sub>.<sub>3980</sub>.<sub>4070</sub>.<sub>4100</sub>.<sub>406</sub><sup>0</sup>.<sup>4070</sup>.<sup>5130</sup>.<sup>4950</sup>.<sup>4190</sup>.<sup>4190</sup>.<sup>4000</sup>.<sup>4060</sup>. <sub>0</sub>.<sub>4480</sub>.<sub>4540</sub>.<sub>4410</sub>.4<sup>910</sup>.<sup>4590</sup>.<sup>4620</sup>.<sup>4490</sup>.<sup>6660</sup>.<sup>5890</sup>.<sup>4870</sup>.<sup>4610</sup>.<sup>4780</sup>.<sup>4500</sup>. <sub>4130</sub>.<sub>3900</sub>.<sub>4040</sub>.<sub>4260</sub>.<sub>4200</sub>.4<sup>210</sup>.<sup>4140</sup>.<sup>5320</sup>.<sup>5150</sup>.<sup>4280</sup>.<sup>4250</sup>.<sup>4100</sup>.<sup>4110</sup>.<sup>4</sup> <sub>T</sub>m<sup>1</sup> <sup>336</sup> <sub>0</sub>.<sub>3900</sub>.<sub>3610</sub>.<sub>3930</sub>.<sub>4040</sub>.<sub>3930</sub>.<sub>3870</sub>.<sub>40</sub>4<sup>0</sup>.<sup>4500</sup>.<sup>4510</sup>.<sup>3980</sup>.<sup>4040</sup>.<sup>3740</sup>.<sup>3870</sup>. <sub>0</sub>.<sub>3680</sub>.<sub>3200</sub>.<sub>3570</sub>.<sub>3340</sub>.<sub>3680</sub>.<sub>3520</sub>.<sub>374</sub><sup>0</sup>.<sup>4040</sup>.<sup>4260</sup>.<sup>3640</sup>.<sup>3870</sup>.<sup>3380</sup>.<sup>3750</sup>.

<sub>0</sub>.<sub>3320</sub>.<sub>2750</sub>.<sub>3230</sub>.<sub>2880</sub>.<sub>3320</sub>.<sub>2900</sub>.<sub>33</sub>4<sup>0</sup>.<sup>7570</sup>.<sup>6100</sup>.<sup>3580</sup>.<sup>4040</sup>.<sup>2910</sup>.<sup>3330</sup>. <sub>4060.3910.3920</sub>.4<sup>070</sup>.<sup>4070</sup>.<sup>4120</sup>.<sup>4041</sup>.<sup>7301</sup>.<sup>0420</sup>.<sup>5580</sup>.<sup>5240</sup>.<sup>4080</sup>.<sup>4030</sup>.<sup>5</sup> <sub>2</sub><sup>0</sup> <sub>0</sub>.<sub>3120</sub>.<sub>3490</sub>.<sub>2980</sub>.<sub>3400</sub>.<sub>31</sub><sup>10</sup>.<sup>3480</sup>.<sup>3090</sup>.<sup>3470</sup>.<sup>5970</sup>.<sup>5420</sup>.<sup>3770</sup>.<sup>4220</sup>.<sup>3210</sup>.<sup>3310</sup> <sub>00</sub>.<sub>3090</sub>.<sub>2370</sub>.<sub>2990</sub>.<sub>2500</sub>.<sub>3090</sub>.<sub>2550</sub>.<sub>3140</sub>.<sub>4</sub>1<sup>40</sup>.<sup>4920</sup>.<sup>2900</sup>.<sup>3640</sup>.<sup>2490</sup>.<sup>3090</sup>.<sup>2</sup> <sub>0</sub>.<sub>2630</sub>.<sub>175</sub>0.<sup>2580</sup>.<sup>1800</sup>.<sup>2640</sup>.<sup>1830</sup>.<sup>2700</sup>.<sup>2870</sup>.<sup>3660</sup>.<sup>2070</sup>.<sup>3050</sup>.<sup>1870</sup>.<sup>2670</sup>.

## 5.2 Main Results

Table 3 presents a comprehensive quantitative evaluation of ASGMamba against ten state-of-the-art baselines across nine diverse real-world datasets. The experiments cover four distinct prediction horizons $T \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ }, providing a rigorous assessment of Long-term Time Series Forecasting (LTSF) capabilities. As evidenced by the aggregated results, ASGMamba achieves the lowest average Mean Squared Error (MSE) on five out of the nine datasets and the lowest average Mean Absolute Error (MAE) on seven out of the nine datasets. This positions ASGMamba as efective in linear-complexity forecasting.

Superiority in Volatile Environments. ASGMamba outperforms advanced Transformer-based architectures on datasets with low Signal-to-Noise Ratios (SNR). For instance, on the Exchange dataset, which features severe structural breaks, ASG-Mamba records an average MSE of 0.351 outperforming iTransformer (0.360) and PatchTST (0.367). Similarly, on the Ettm1 dataset, ASGMamba surpasses both the vanilla S-Mamba (0.398) and TimeMixer (0.381) with an MSE of 0.376. These results validate our core hypothesis: in scenarios laden with stochastic perturbations, the proposed Adaptive Spectral Gating mechanism efectively filters high-frequency noise, allowing the model to capture robust long-term trends that standard attention mechanisms often miss due to overfitting.

Limitations on High-Dimensional Data. However, the performance landscape shifts on high-dimensional datasets with massive channel counts, such as Trafic (862 variates) and Electricity (321 variates). While ASGMamba remains competitive against PatchTST, it yields suboptimal results compared to iTransformer and the vanilla S-Mamba. On Trafic, ASGMamba achieves an average MSE of 0.478, trailing behind S-Mamba (0.414). We attribute this to the aggressive nature of our gating mechanism. Trafic data is characterized by high-entropy, non-stationary fluctuations that are physically meaningful (e.g., sudden congestion) but spectrally resemble noise. The gating mechanism may inadvertently suppress these high-frequency signals, leading to information loss. Furthermore, the vanilla S-Mamba utilizes a bidirectional scanning mechanism across variates, which naturally excels at capturing dense spatial correlations in high-dimensional settings, whereas our explicit Node Embedding strategy may face capacity bottlenecks when scaling to nearly a thousand sensors.

Mechanism Validation and Stability. Despite the trade-of in high-dimensional settings, ASGMamba demonstrates superior stability on the ETT family datasets compared to the baseline S-Mamba. On ETTh1 and ETTm1, our model improves upon S-Mamba by approximately 3.1% and 1.3% respectively. This confirms that for datasets where variables operate with distinct physical semantics but moderate dimensionality, the combination of spectral gating and semantic node embeddings provides a clear advantage over raw state space modeling. Additionally, ASGMamba exhibits exceptional robustness at the ultra-long horizon (T = 720). For example, on ETTm2, it achieves an MSE of 0.401, matching the performance of specialized decomposition models like TimeMixer and significantly outperforming older baselines like FEDformer (0.421). In summary, ASGMamba ofers a specialized trade-of: it sacrifices some granularity on massive-channel datasets to achieve strong noise robustness and trend capture capability in volatile, low-SNR environments.

![](images/fig2.jpg)  
(a) Etth2-96-ASGMamba

![](images/fig3.jpg)  
(d) Etth2-96-SMamba

![](images/fig4.jpg)  
(c) Etth2-96-TimeMixer

![](images/fig5.jpg)  
(b) Etth2-96-iTransformer

![](images/fig6.jpg)  
(e) Etth2-96-CrossFormer

![](images/fig7.jpg)  
(f) Etth2-96-PatchTST

![](images/fig8.jpg)  
(g) Etth2-96-TimesNet

![](images/fig9.jpg)  
(h) Etth2-96-Dlinear  
Fig. 2 Visual comparison of forecasting performance on the Etth2 dataset with a horizon of $T = 9 6$

## 5.3 Visualizing and Analyzing

To intuitively evaluate the temporal modeling capability of ASGMamba, we visualize the forecasting results on the ETTh2 dataset with a prediction horizon of T = 96. Figure 2 presents a qualitative comparison between ASGMamba and seven representative baselines, including S-Mamba, iTransformer, and PatchTST. The ETTh2 dataset is characterized by strong cyclicity mixed with irregular amplitude shifts and local fluctuations, posing a dual challenge: models must capture the dominant seasonal trend without overfitting to high-frequency stochastic noise or lagging in phase during rapid transitions.

As illustrated in Fig. 2(a), ASGMamba demonstrates superior alignment with the Ground Truth curve (orange). Specifically, it accurately reconstructs the sharp peaks and troughs of the load cycle with minimal phase shift. In contrast, the vanilla S-Mamba (Fig. 2(d)) and linear baselines like DLinear (Fig. 2(h)) exhibit noticeable amplitude decay and ”smoothing efects” as the prediction steps increase, failing to capture the full intensity of peak loads. While advanced Transformers like PatchTST (Fig. 2(f)) and iTransformer (Fig. 2(b)) successfully model the general trend, they show slight deviations in local details and instability at the boundaries of the prediction window compared to our method.

The visual superiority supports the theoretical benefits of the proposed Adaptive Spectral Gating mechanism. By analyzing local spectral energy, ASGMamba efectively filters out unnecessary high-frequency noise before it enters the state space. This denoising process enables the selective scan mechanism to focus its limited capacity on preserving strong periodic patterns, rather than fitting noise. As a result, ASG-Mamba produces forecasts that are not only statistically accurate but also consistent with the underlying dynamic system, avoiding the spurious fluctuations commonly seen in time-domain models.

## 5.4 Ablation Studies

To disentangle the contribution of individual components within ASGMamba and validate our design choices, we conducted a comprehensive ablation analysis. We selected ETTh1 and Weather as representative benchmarks due to their contrasting statistical properties: ETTh1 is characterized by strong seasonality corrupted by noise, whereas Weather exhibits complex, non-linear dynamics with high volatility. We evaluated the full model against four variants, systematically isolating the Adaptive Spectral Gating, Multi-Scale Fusion, Overlapping Patching, and Gating Strategies. The quantitative results are presented in Table 4.

## 5.4.1 Impact of Adaptive Spectral Gating

The Spectral Gating structurally distinguish robust periodic patterns from high frequency stochastic noise. In the w/o Spectral Gating variant, we bypassed this mechanism (setting gating weights G ≡ 1), efectively reverting the backbone to a standard Mamba block.

As shown in Table 4, removing FAR brings about certain performance drop. Specif ically, we find that MSE increases about 3.45% on ETTh1 and 2.83% on Weather on average. Such uniform drop verifies our hypothesis on standard SSMs limitation: without spectral guidance, the selective scan mechanism treats input tokens as an entire pool. For datasets containing abundant stochastic perturbation $( \mathrm { e . g . }$ , Weather), to fit high-frequency noise, the model misuses precious state capacity. By introducing FAR, ASGMamba plays the role of an adaptive spectral filter that suppresses noisedominated components in the frequency domain from invading state space before they pollute unseen future horizons and improve generalization.

<sub>rformance</sub> <sub>dete</sub>r<sup>ioration</sup> <sup>(%)</sup> <sup>for</sup> <sup>each</sup> <sup>abl</sup> Table4AblationresultsfordiferentASGMambamodulesonETTh1andWeather.ThetabledisplaysMSEandMAEvalues,aswellasthe

<table><tr><td colspan="2">Model Variant</td><td colspan="2">Ours</td><td colspan="2">w/o Spectral Gating</td><td colspan="2">w/o Multi-Scale Fusion</td><td colspan="2">w/o Patching</td><td colspan="2">w/o Gating</td></tr><tr><td>Metric</td><td></td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.161</td><td>0.207</td><td>0.169</td><td>0.221</td><td>0.169</td><td>0.218</td><td>0.166</td><td>0.213</td><td>0.165</td><td>0.215</td></tr><tr><td>192</td><td>0.206</td><td>0.246</td><td>0.212</td><td>0.252</td><td>0.216</td><td>0.254</td><td>0.212</td><td>0.251</td><td>0.209</td><td>0.252</td></tr><tr><td>336</td><td>0.265</td><td>0.291</td><td>0.268</td><td>0.301</td><td>0.275</td><td>0.305</td><td>0.272</td><td>0.301</td><td>0.273</td><td>0.298</td></tr><tr><td>720</td><td>0.344</td><td>0.339</td><td>0.352</td><td>0.343</td><td>0.356</td><td>0.353</td><td>0.346</td><td>0.350</td><td>0.348</td><td>0.346</td></tr><tr><td>Degradation</td><td></td><td>-</td><td>-</td><td>2.83%</td><td>3.45%</td><td>4.27 %</td><td>4.38 %</td><td>2.31%</td><td>2.90 %</td><td>2.03 %</td><td>2.69 %</td></tr><tr><td rowspan="4">Etth1</td><td>96</td><td>0.369</td><td>0.391</td><td>0.382</td><td>0.402</td><td>0.392</td><td>0.406</td><td>0.381</td><td>0.393</td><td>0.381</td><td>0.398</td></tr><tr><td>192</td><td>0.426</td><td>0.419</td><td>0.438</td><td>0.432</td><td>0.439</td><td>0.442</td><td>0.436</td><td>0.435</td><td>0.438</td><td>0.426</td></tr><tr><td>336</td><td>0.478</td><td>0.444</td><td>0.499</td><td>0.456</td><td>0.496</td><td>0.468</td><td>0.488</td><td>0.453</td><td>0.489</td><td>0.456</td></tr><tr><td>720</td><td>0.487</td><td>0.473</td><td>0.502</td><td>0.495</td><td>0.516</td><td>0.496</td><td>0.493</td><td>0.482</td><td>0.496</td><td>0.486</td></tr><tr><td>Degradation</td><td></td><td>-</td><td>-</td><td>3.45 %</td><td>3.32 %</td><td>4.75 %</td><td>4.90%</td><td>2.23%</td><td>2.06%</td><td>2.56 %</td><td>2.16%</td></tr></table>

## 5.4.2 Eficacy of Adaptive Multi-Scale Fusion

Semantic temporal patterns naturally appear at diferent granularities, which present a dilemma between local resolution and global context. ASGMamba alleviates this by designing a three-branch architecture $( P \in \{ 8 , 1 6 , 3 2 \} )$ ). w/o Multi-Scale Fusion is forced to rely on only one scale $( P = 1 6 )$ . The accuracy drop is significant, which is even larger than that caused by other modules. That is, we can observe an approximate 4.75% MSE increase on ETTh1 and 4.27% on Weather. We attribute this to the fixed receptive field of single-scale-patches. Specifically, small patches $( P = 8 )$ capture fine grained jitter while they lack global context for long-term dependency modeling; while large patches $( P = 3 2 )$ capture global trend but sufer from spectral smoothing, leading to loss of high-frequency information. ASGMamba adapts to dynamically aggregating these two views. The model can focus on intrinsic periodicity and volatility of diferent variables and assign diferent weights.

## 5.4.3 Significance of Overlapping Patching

Conventional non-overlapping patching (Stride = Patch) introduces artificial ”boundary artifacts” disrupting the semantic continuity of local temporal patterns. We validated the proposed Overlapping Patching strategy $\mathrm { ( S t r i d e = P a t c h / 2 ) }$ against the non-overlapping baseline.

While the performance gain (approx. 2.23% on ETTh1) is more modest compared to the Multi-Scale architecture, the consistent improvement across datasets underscores its necessity. By enforcing a 50% overlap, we ensure that critical transition points are encoded in multiple patches with varying relative positions. This redundancy provides the Mamba encoder with enriched contextual views, mitigating information loss at patch boundaries and fostering smoother latent state transitions.

## 5.4.4 Sensitivity to Gating Strategies

We assess whether gains stem from the spectral domain by comparing variants. Specifically, we benchmark our spectral gating against a non-gated baseline (Identity) and a variant devoid of frequency-aware capabilities.

Empirically, the $\mathrm { w / o }$ Gating variant yielded a smaller degradation (approx. 2.0% on Weather) compared to the removal of the Adaptive Spectral Gating logic (2.83%), while the full ASGMamba achieves the lowest MSE (e.g., 0.369 on ETTh1-96). This disparity highlights the fundamental advantage of spectral analysis in signal separation. In the time domain (or simple identity mapping), signal and noise are intricately entangled, making it dificult to distinguish a valid anomaly from a random spike. In

---

contrast, stochastic noise typically exhibits a distinct ”broadband” energy signature in the frequency domain. This spectral distinctiveness enables the Gating Mechanism to identify and suppress noise components more robustly than standard gating approaches, justifying the coupling of FFT with the gating mechanism.

## 5.5 Hyper-parameter Sensitivity Analysis

To verify the robustness of ASGMamba and identify the optimal configuration for forecasting tasks, we conducted a comprehensive sensitivity analysis on two critical hyper-parameters: SSM State Dimension (N) and the Spectral Bands $( K _ { f r e q } )$ . All experiments were performed on the Weather dataset with a prediction horizon of $T = 9 6$

Impact of SSM State Dimension The state dimension N controls the memory capacity of the selective scan mechanism. We tested $N \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ . The results in Fig. 3(a) demonstrate that ASGMamba is relatively robust to N. While $N = 8$ underperforms due to constrained information bottlenecks, $N = 1 6$ achieves competitive results comparable to larger settings $( N = 3 2 , 6 4 )$ . This suggests that the Adaptive Spectral Gating eficiently filters noise, allowing the SSM to store compact, robust dynamics without requiring an excessively large state space.

![](images/fig10.jpg)

![](images/fig11.jpg)  
(a) Impact of SSM state dimension N. (b) Impact of spectral band granularity $K _ { f r e q }$ Fig. 3 Hyperparameter sensitivity analysis on the Weather dataset.

## Impact of Spectral Band Granularity

The number of spectral bands into which the spectrum is divided is equal to the resolution of the noise filter. Therefore, we also tried to divide the spectrum into $K _ { f r e q } \in \{ 1 , 2 , 3 , 4 \}$ bands. As illustrated in Fig. 3(b), the case of $K _ { f r e q } = 1$ (Global Energy) is suboptimal because the gating mechanism cannot distinguish high energy informative signals from high energy artifacts. The accuracy improves until $K _ { f r e q } =$ 3. Physically, this division matches the common sense of time series decomposition. Indeed, the spectrum is properly separated into three parts: Trend $\left( \mathrm { L o w } \right)$ , Periodicity (Mid), and Noise (High). When $K _ { f r e q } = 4$ , there is an additional cost without any improvement in performance, which confirms that $K _ { f r e q } = 3$ is the optimal setting.

![](images/fig12.jpg)  
Fig. 4 Eficiency comparison on the weather dataset $( L = 9 6 , T = 1 9 2 )$ . The x-axis denotes training speed (s/epoch), the y-axis denotes MSE, and the bubble area represents GPU memory usage. ASGMamba (Red) achieves the optimal trade-of.

![](images/fig13.jpg)  
Fig. 5 Eficiency comparison on the ETTm1 dataset $( L = 9 6 , T = 1 9 2 )$ . The x-axis denotes training speed (s/epoch), the y-axis denotes MSE, and the bubble area represents GPU memory usage. ASGMamba (Red) achieves the optimal trade-of.

## 5.6 Eficiency Analysis

We evaluate the practical utility of forecasting models by analyzing the trade-of between predictive accuracy and computational cost. In Fig. 4 and Fig. 5, we plot a high-dimensional performance space where x-axis denotes training speed, y-axis denotes forecasting error (MSE), and the bubble size denotes GPU memory footprint. Ideally, a model should be located in the bottom-left corner (Pareto frontier). Since we are interested in lightweight linear models are ruled out, high-capacity models are plotted. They are capable of modeling non-linear dynamics. ASGMamba (red bubble) is always able to achieve the best performance with least memory footprint. Unlike Transformer-based architectures, ASGMamba shows a clear advantage due to the high memory cost of quadratic attention complexity $\mathcal { O } ( L ^ { 2 } )$ . For example, on Weather dataset, Crossformer needs 5.18GB memory and 81s/epoch due to its two-stage attention mechanism. While ASGMamba only needs 0.62GB memory (reduced by 8×) and 9s/epoch (9× speedup) while achieving lower MSE. The linear-complexity spectral gating indeed removes the redundancy existing in deep attention networks. In addition, ASGMamba is significantly faster/finer than eficient CNN/MLP-based baselines. ASGMamba is approximately 4× faster than TimesNet due to the heavy 2D convolutions used in TimesNet for temporal modeling. Unlike TimeMixer, average pooling will drop the high-frequency details of the signal. While our orthogonal wavelet decomposition model preserves the details of the signal. Therefore, ASGMamba achieves higher accuracy without latency overhead, making it more applicable to real-world scenarios with limited resources.

## 6 Conclusion

In this work, we introduce ASGMamba, a forecasting framework that addresses the short-state-capacity issue of linear SSMs and the challenging statistical properties of real-world time series spectral characteristics. We lift this limitation by designing an Adaptive Spectral Gating mechanism that lets the state evolve in a spectralconditioned manner, enabling the filtering of broadband noise injected at the input stage, while avoiding the associated cost of performing global frequency transformations. The proposed architecture maintains the conservative O(L) scaling of its Mamba backbone while keeping enhancing a great robustness to volatile dynamics.

Empirical evaluations across nine diverse benchmarks demonstrate that ASG-Mamba achieves a Pareto-optimal trade-of: it matches or exceeds the predictive accuracy of computationally intensive Transformer architectures while delivering high inference throughput and a minimal memory footprint. These attributes establish ASGMamba as a scalable solution particularly well-suited for high-frequency forecasting tasks in resource-constrained or latency-sensitive computational environments.

Future investigations will focus on extending this spectral-aware state space formulation to natively accommodate irregularly sampled or asynchronous time series. Furthermore, given its linear eficiency, we intend to explore ASGMamba as a token-eficient backbone for large-scale pre-trained time series foundation models.

Acknowledgments This work is supported by the National Natural Science Foundation of China (No. 62372366).

Data availability Data is available on request.

## Declarations

Conflict of interest The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

[1] Liu, X., Xia, Y., Liang, Y., Hu, J., Wang, Y., Bai, L., Huang, C., Liu, Z., Hooi, B., Zimmermann, R.: Largest: A benchmark dataset for large-scale trafic forecasting. Advances in Neural Information Processing Systems 36, 75354–75371 (2023)

[2] Sezer, O.B., Gudelek, M.U., Ozbayoglu, A.M.: Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied soft computing 90, 106181 (2020)

[3] Arsenault, P., Wang, S., Patenaude, J.: A survey of explainable artificial intelligence (XAI) in financial time series forecasting. ACM Comput. Surv. 57(10), 265–126537 (2025) https://doi.org/10.1145/3729531

[4] Deb, C., Zhang, F., Yang, J., Lee, S.E., Shah, K.W.: A review on time series forecasting techniques for building energy consumption. Renewable and Sustainable Energy Reviews 74, 902–924 (2017)

[5] Lago, J., Marcjasz, G., De Schutter, B., Weron, R.: Forecasting day-ahead elec tricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. Applied Energy 293, 116983 (2021) https://doi.org/10. 1016/j.apenergy.2021.116983

[6] Wang, Y., Wu, H., Dong, J., Liu, Y., Long, M., Wang, J.: Deep time series models: A comprehensive survey and benchmark. arXiv preprint arXiv:2407.13278 (2024)

[7] Yu, Q., Yang, G., Wang, X., Shi, Y., Feng, Y., Liu, A.: A review of time series forecasting and spatio-temporal series forecasting in deep learning. J. Supercomput. 81(10), 1160 (2025) https://doi.org/10.1007/S11227-025-07632-W

[8] Kim, J., Kim, H., Kim, H., Lee, D., Yoon, S.: A comprehensive survey of deep learning for time series forecasting: architectural diversity and open challenges. Artificial Intelligence Review 58(7), 1–95 (2025)

[9] Qiu, X., Hu, J., Zhou, L., Wu, X., Du, J., Zhang, B., Guo, C., Zhou, A., Jensen, C.S., Sheng, Z., Yang, B.: Tfb: Towards comprehensive and fair benchmarking of time series forecasting methods. Proc. VLDB Endow. 17(9), 2363–2377 (2024)

[10] Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J.: A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730 (2022)

[11] Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., Long, M.: itransformer: Inverted transformers are efective for time series forecasting. arXiv preprint arXiv:2310.06625 (2023)

[12] Zhao, J., Chu, F., Xie, L., Che, Y., Wu, Y., Burke, A.F.: A survey of transformer networks for time series forecasting. Comput. Sci. Rev. 60, 100883 (2026) https: //doi.org/10.1016/J.COSREV.2025.100883

[13] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., Zhang, W.: Informer: Beyond eficient transformer for long sequence time-series forecasting. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 11106–11115 (2021)

[14] Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. CoRR abs/2312.00752 (2023) https://doi.org/10.48550/ARXIV.2312. 00752 2312.00752

[15] Aoki, M.: State Space Modeling of Time Series. Springer, Berlin, Heidelberg (2013)

[16] Gu, A., Goel, K., R´e, C.: Eficiently modeling long sequences with structured state spaces. In: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, Online (2022). https://openreview.net/forum?id=uYLFoz1vlAC

[17] Ghil, M., Allen, M.R., Dettinger, M.D., Ide, K., Kondrashov, D., Mann, M.E., Robertson, A.W., Saunders, A., Tian, Y., Varadi, F., et al.: Advanced spectral methods for climatic time series. Reviews of geophysics 40(1), 3–1 (2002)

[18] Qiu, X., Cheng, H., Wu, X., Hu, J., Guo, C.: A comprehensive survey of deep learning for multivariate time series forecasting: A channel strategy perspective. CoRR abs/2502.10721 (2025) https://doi.org/10.48550/ARXIV.2502. 10721 2502.10721

[19] Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., Jin, R.: Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In: International Conference on Machine Learning, pp. 27268–27286 (2022). PMLR

[20] Chen, W., Ye, J., Zhao, C., Huang, Y.: MFFCNN: multi-scale fractional fourier transform convolutional neural network for multivariate time series forecasting. J. Supercomput. 81(2), 416 (2025) https://doi.org/10.1007/S11227-024-06888-Y

[21] Taylor, S.J., Letham, B.: Forecasting at scale. The American Statistician 72(1),

[22] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

[23] Wu, H., Xu, J., Wang, J., Long, M.: Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems 34, 22419–22430 (2021)

[24] Zhou, C., Jiang, K., Liu, Y., Che, C., Zhang, Q.: MSTF: enhancing long term forecasting with multi-scale temporal fusion in time series forecasting. J. Supercomput. 81(9), 1082 (2025) https://doi.org/10.1007/S11227-025-07572-5

[25] Somvanshi, S., Islam, M.M., Mimi, M.S., Polock, S.B.B., Chhetri, G., Das, S.: From S4 to Mamba: A Comprehensive Survey on Structured State Space Models (2025). https://arxiv.org/abs/2503.18970

[26] Nguyen, E., Goel, K., Gu, A., Downs, G.W., Shah, P., Dao, T., Baccus, S., R´e, C.: S4ND: modeling images and videos as multidimensional signals with state spaces. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A. (eds.) Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Online (2022). http://papers.nips.cc/paper files/paper/2022/hash/13388efc819c09564c66ab2dc8463809- Abstract-Conference.html

[27] Ahamed, M.A., Cheng, Q.S.: Timemachine: A time series is worth 4 mambas for long-term forecasting. In: Endriss, U., Melo, F.S., Bach, K., Diz, A.J.B., Alonso-Moral, J.M., Barro, S., Heintz, F. (eds.) ECAI 2024 - 27th European Conference on Artificial Intelligence, 19-24 October 2024, Santiago de Compostela, Spain - Including 13th Conference on Prestigious Applications of Intelligent Systems (PAIS 2024). Frontiers in Artificial Intelligence and Applications, vol. 392, pp. 1688–1695. IOS Press, Online (2024). https://doi.org/10.3233/FAIA240677 . https://doi.org/10.3233/FAIA240677

[28] Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., Long, M.: Timesnet: Temporal 2d-variation modeling for general time series analysis. In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, Online (2023). https://openreview.net/forum?id=ju Uqw384Oq

[29] Xu, Z., Zeng, A., Xu, Q.: FITS: modeling time series with 10k parameters. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, ??? (2024). https://openreview.net/forum?id=bWcnvZ3qMb

[30] Bao, J., Tian, J., Jia, X., Huang, Y.: Msfmoe: A multi-scale frequency filtering network with mixture of experts for time series forecasting. Neurocomputing 668, 132414 (2026) https://doi.org/10.1016/J.NEUCOM.2025.132414

[31] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., Choo, J.: Reversible instance normalization for accurate time-series forecasting against distribution shift. In: International Conference on Learning Representations (2021)

[32] Wang, Z., Kong, F., Feng, S., Wang, M., Yang, X., Zhao, H., Wang, D., Zhang, Y.: Is mamba efective for time series forecasting? Neurocomputing 619, 129178 (2025) https://doi.org/10.1016/J.NEUCOM.2024.129178

[33] Zhang, Y., Yan, J.: Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In: The Eleventh International Conference on Learning Representations (2023)

[34] Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y., ZHOU, J.: Timemixer: Decomposable multiscale mixing for time series forecasting. In: The Twelfth International Conference on Learning Representations (2024). https://openreview.net/forum?id=7oLshfEIC2

[35] Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., Yu, R.: Long-term forecasting with tide: Time-series dense encoder. Trans. Mach. Learn. Res. 2023 (2023)

[36] Zeng, A., Chen, M., Zhang, L., Xu, Q.: Are transformers efective for time series forecasting? In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, pp. 11121–11128 (2023)