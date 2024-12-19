# Improving State Space Models for Computer Vision
> CSCI-GA.2271-001 (Advanced) Computer Vision - Fall 2024

This project explores techniques with the potential of improving state-space models (SSMs) for computer vision tasks, focusing on lightweight architectures and efficient training methods. Our work builds upon prior research, including the original **Vision Mamba (Vim)** [[1](https://arxiv.org/abs/2401.09417)] and **MambaVision** [[2](https://arxiv.org/abs/2407.08083)], and extends these architectures with optimization techniques and experiments involving novel integrations such as Squeeze-and-Excitation (SE) blocks, Early-Bird Lottery Ticket Hypothesis, Sliding Window Attention (SWA), and more!

## Repository Structure

This repository is organized into two main folders:

- **`vision_mamba/`**  
  Contains the implementation, experiments, and checkpoints for Vision Mamba (Vim).

- **`mamba_vision/`**  
  Includes the MambaVision, the hybrid Vision Transformer and Mamba-based architecture, along with its checkpoints and experiments.

### Subfolder: `checkpoints/`
Both the `vision_mamba/` and `mamba_vision/` folders include a `checkpoints/` directory to store trained models and weights for reproducibility.

---

## Experiments

The experiments conducted in this project are organized into two categories, corresponding to Vision Mamba (Vim) and MambaVision. Each experiment is located in its respective folder (`vision_mamba/` or `mamba_vision/`) and includes some checkpoints for reproducibility. Below is an overview of the experiments performed:

### With Vision Mamba (Vim):
- **Squeeze-and-Excitation (SE) Integration** [[3](https://arxiv.org/abs/1709.01507)]: Evaluated SE block integration to recalibrate features, achieving variable results.
- **MobileNetV2-Inspired Depthwise Convolutions** [[4](https://arxiv.org/abs/1801.04381)]: Reduces computational overhead with slight accuracy variation.
- **Dynamic Pruning with Cosine Scheduler**: Improved sparsity efficiency while maintaining high accuracy, achieving a best accuracy of 70%.
- **Iterative Magnitude Pruning** [[5](https://arxiv.org/abs/2210.03044)]: Repeatedly pruned low-magnitude weights to identify sparse, efficient subnetworks while retaining performance.
- **Early-Bird Lottery Ticket Hypothesis** [[6](https://arxiv.org/abs/1909.11957)]: Identified sparse subnetworks within the first 6 epochs by stabilizing mask distances, enabling early optimization.
- **Gradient Pruning**: Pruned weights based on gradient magnitudes during training, allowing for adaptive sparsity that targets the most insignificant weights while maintaining performance.


### With MambaVision:
- **Sliding Window Attention (SWA)**: Investigated increased receptive fields for MambaVision on larger-scale tasks.
- **Exploration of S4/S5 SSMs**: Tested alternative kernels (S4 and S5) for continuous signals, showing promise in handling vision tasks with continuous characteristics.

---

## References
1. L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and
X. Wang, “Vision mamba: Efficient visual representation learning
with bidirectional state space model,” 2024. [Online]. Available:
https://arxiv.org/abs/2401.09417
2. A. Hatamizadeh and J. Kautz, “Mambavision: A hybrid
mamba-transformer vision backbone,” 2024. [Online]. Available:
https://arxiv.org/abs/2407.08083
3. J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, “Squeeze-and-excitation
networks,” 2019. [Online]. Available: https://arxiv.org/abs/1709.01507
4. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen,
“Mobilenetv2: Inverted residuals and linear bottlenecks,” 2019. [Online].
Available: https://arxiv.org/abs/1801.04381
5. M. Paul, F. Chen, B. W. Larsen, J. Frankle, S. Ganguli, and
G. K. Dziugaite, “Unmasking the lottery ticket hypothesis: What’s
encoded in a winning ticket’s mask?” 2022. [Online]. Available:
https://arxiv.org/abs/2210.03044
6. H. You, C. Li, P. Xu, Y. Fu, Y. Wang, X. Chen, R. G. Baraniuk,
Z. Wang, and Y. Lin, “Drawing early-bird tickets: Towards more
efficient training of deep networks,” 2022. [Online]. Available:
https://arxiv.org/abs/1909.11957