# Towards the Control Theory of ANNs
How to control ANNs within several seconds.


# Based on What We Can Control Artificial Neural Networks [arxiv](https://arxiv.org/abs/2310.05692)

> [**Based on What We Can Control Artificial Neural Networks**]()

## 📰 News
[2024.05.09] Fist release of code. 
[2023.10.09] Code will be released in a few days (not too long). Please stay tuned or *watch this repo* for quick information.



## 💗 Highlights
Based on the knowledge of control systems, designing proper optimisers (or controllers) and advanced learning systems can benefit the learning process and complete relevant tasks (e.g., classification and generation). In this paper, we design two advanced optimisers and analyze three learning systems relying on the control system knowledge. The contributions are as follows: 

### 🔥 Optimisers are controllers
<font face="Black" size="4">(1)</font> PID and  SGDM (PI controller) optimiser performs more stable than SGD (P controller), SGDM (PI controller), AdaM and fuzzyPID optimisers on most residual connection used CNN models. <font face="Black" size="4">(2)</font> HPF-SGD outperforms SGD and LPF-SGD, which indicates that high frequency part is significant during SGD learning process. <font face="Black" size="4">(3)</font> AdaM is an adaptive filter that combines an adaptive filter and an accumulation adaptive part.


### 🔥🔥 Learning systems of most ANNs are control systems
<font face="Black" size="4">(1)</font> Most ANNs present perfect consistent performance with their system response. <font face="Black" size="4">(2)</font> We can use proper optimisers to control and improve the learning process of most ANNs. 






### 🔥🔥🔥 The Optimiser should Match the Learning System
<font face="Black" size="4">(1)</font> RSs based vision models prefer SGDM, PID and fuzzyPID optimisers. <font face="Black" size="4">(2)</font> RS mechanism is similar to AdaM. particularly, SGDM optimizes the weight of models on the time dimension, and RS optimizes the model on the space dimension. <font face="Black" size="4">(3)</font> AdaM significantly benefits FFNN and GAN, but PID and FuzzyPID dotes CycleGAN most. 


