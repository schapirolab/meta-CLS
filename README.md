
This repository provides the implementation of our paper  
**A Gradient of Complementary Learning Systems Emerges Through Meta‑Learning**  

It builds upon the official [La‑MAML](https://github.com/montrealrobotics/La-MAML) code by Gupta et al. (NeurIPS 2020), extending it to meta‑learn layer‑wise learning rates and representational sparsity in both single‑ and dual‑pathway artificial neural networks for continual learning tasks.


## Data Preparation

1. Create a `data/` directory in the repository root.  
2. Download and place the following datasets into `data/` before running experiments:  
   - rotated MNIST  
   - Fashion MNIST  
   - CIFAR‑100  

---

## Running the Experiments

Example: make scripts executable  
   ```bash
   chmod +x run_experiments_2layer_models    # for shallow network experiments
   ```
---


## Citation

If you use this code, please cite both the original La‑MAML paper and our extension:

```bibtex
@article{gupta2020,
  title={La-MAML: Look-ahead Meta Learning for Continual Learning},
  author={Gupta, Gunshi and Yadav, Karmesh and Paull, Liam},
  journal={arXiv preprint arXiv:2007.13904},
  year={2020}
}

@article{zhou2025,
  title={A gradient of complementary learning systems emerges through meta-learning},
  author={Zhou, Zhenglong and Schapiro, Anna},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.07.10.664201}
}
```

---


