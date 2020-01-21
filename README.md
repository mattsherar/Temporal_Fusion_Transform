# Temporal_Fusion_Transform
Pytorch Implementation of Google's TFT

Original Github link: https://github.com/google-research/google-research/tree/master/tft

Paper link: https://arxiv.org/pdf/1912.09363.pdf

Abstract
Multi-horizon forecasting problems often contain a complex mix of inputs -- including static (i.e. time-invariant) covariates, known future inputs, and other exogenous time series that are only observed historically -- without any prior information on how they interact with the target. While several deep learning models have been proposed for multi-step prediction, they typically comprise black-box models which do not account for the full range of inputs present in common scenarios. In this paper, we introduce the Temporal Fusion Transformer (TFT) -- a novel attention-based architecture which combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics. To learn temporal relationships at different scales, the TFT utilizes recurrent layers for local processing and interpretable self-attention layers for learning long-term dependencies. The TFT also uses specialized components for the judicious selection of relevant features and a series of gating layers to suppress unnecessary components, enabling high performance in a wide range of regimes. On a variety of real-world datasets, we demonstrate significant performance improvements over existing benchmarks, and showcase three practical interpretability use-cases of TFT.
