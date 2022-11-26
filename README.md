# Symbolic Deep Learning for Structural System Identification
Abstract:

Closed-form model expression is commonly required for parametric data assimilation (e.g., model updating, damage quantification, and so on). However, epistemic bias due to fixing the model class is a challenging issue for structural identification. Furthermore, it is sometimes hard to derive explicit expressions for structural mechanisms such as damping and nonlinear restoring forces. Although existing model class selection methods are beneficial to reduce the model uncertainty, the primary issue lies in their limitation to a small number of predefined model choices. We propose a symbolic deep learning framework that alleviates the constraint of fixed model classes and lets the data more flexibly determine the model type and discover the symbolic invariance of the structural system. A design principle for symbolic neural networks has been developed to leverage domain knowledge and translate data to flexibly symbolic equations of motion with a good predictive capacity for new data. A two-stage model selection strategy is proposed to conduct adaptive pruning on network and equation levels by balancing the model sparsity and the goodness of fit. The proposed method’s expressive strengths and weaknesses have been analyzed in several numerical case studies, including systems with nonlinear damping, restoring force, and chaotic behavior. Results from an experimental case study revealed the potential of the proposed method for flexibly interpreting hidden mechanisms for real-world applications. Finally, we discuss necessary improvements to transfer this computational method for practical applications.

## Citation
<pre>
@article{chen2022symbolic,
  title={Symbolic Deep Learning for Structural System Identification},
  author={Chen, Zhao and Liu, Yang and Sun, Hao},
  journal={Journal of Structural Engineering},
  volume={148},
  number={9},
  pages={04022116},
  year={2022},
  publisher={American Society of Civil Engineers}
}
</pre>
