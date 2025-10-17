# Causal Control TTS

This repository hosts the **demo page** for the *causal control* methodology introduced in the paper: **A Closer Look at Internal Representations of End-to-End Text-to-Speech Models: How is Phonetic and Acoustic Information Encoded?**

📄 **Paper link:** [SSRN – A Closer Look at Internal Representations of End-to-End Text-to-Speech Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217280)

---

## 🎯 Project Overview

**Causal Control TTS** presents interactive audio demonstrations showcasing **causal control**, a novel method enabling **explicit manipulation of acoustic and prosodic features** in neural TTS systems such as **Tacotron2** and **FastSpeech2**.

This work builds upon an in-depth analysis of **how phonetic and acoustic information is encoded** within end-to-end TTS models, bridging **speech science** and **deep learning explainability**.

The repository aims to provide:
- 🎧 Interactive demos illustrating the effects of causal control on synthesized speech  
- 🧠 Insights into how internal representations encode linguistic and acoustic features  
- 💬 Examples comparing controlled vs. uncontrolled synthesis outputs  

---

## 🌐 Demo Page

👉 Explore the live demo here: [**Causal Control TTS Demo**](./demo/index.html)

---

## 🧾 Abstract

> In recent years, deep neural architectures have demonstrated groundbreaking performances in various speech processing areas, including Text-To-Speech (TTS). Models have grown larger, including more layers and millions of trainable parameters to achieve near-natural synthesis, at the expense of interpretability of computed intermediate representations. However, the statistical learning performed by these neural models offers a valuable source of information about language and speech production. The present study aims to develop statistical tools to narrow the gap between these advanced processing techniques and speech sciences. By linearly probing phonetic and acoustic features in model representations, the proposed methods help to understand how neural TTS are able to organize speech information in an unsupervised manner and provide novel insights on phonetic regularities captured through statistical learning on massive datasets that extend beyond human expertise. This study takes a step further by leveraging these insights to design emerging control mechanisms for speech synthesis models, without requiring additional data or training processes. The proposed control is evaluated across a variety of acoustic and prosodic parameters relevant to the perception of speech expressivity. The promising performance of these control mechanisms underscores the value of employing explainability methods in a broader range of domains, enabling neural models to be viewed not merely as tools, but as frameworks that invite a deeper exploration of their underlying mechanisms and structures. Such an approach fosters more comprehensive insights that can improve both the technology and its applications.

---

## 🧩 Citation

If you use this work in your research, please cite:

```
@article{lenglet2023closer,
  title={A Closer Look at Internal Representations of End-to-End Text-to-Speech Models: How is Phonetic and Acoustic Information Encoded?},
  author={Lenglet, Martin and Perrotin, Olivier and Bailly, G{\'e}rard},
  journal={Available at SSRN 5217280},
  year={2023}
}
```

---

## 🧠 Authors and Affiliations

- **Martin Lenglet** – Univ. Grenoble Alpes, CNRS, Grenoble-INP, GIPSA-lab, France / Atos, Échirolles, France  
- **Olivier Perrotin** – Univ. Grenoble Alpes, CNRS, Grenoble-INP, GIPSA-lab, France  
- **Gérard Bailly** – Univ. Grenoble Alpes, CNRS, Grenoble-INP, GIPSA-lab, France  


---

### 🔗 Links

- [Paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217280)  
- [Demo Page](./demo/index.html)

---

© 2025 Martin Lenglet, Olivier Perrotin, Gérard Bailly  
Licensed under CC BY-NC 4.0