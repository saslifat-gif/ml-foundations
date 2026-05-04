# ml-foundations

Implementing the mathematical foundations of machine learning from scratch — one concept at a time.

Each notebook is self-contained: plain Python, minimal dependencies, built to understand not just to run.

---

## Notebooks

###  Probability & Statistics

**`probability-statistics/01_bayes_theorem.ipynb`**

The librarian vs. farmer problem from Kahneman & Tversky. Builds Bayes' theorem from scratch using only base rates and likelihoods, then visualizes the prior vs. posterior shift.

- Conditional probability intuition
- Base rate neglect explained through code
- Matplotlib visualization of belief update

---

###  Flow Matching

**`flow matching/`**

Implementing Flow Matching (Lipman et al. 2022) from scratch on a 2D toy dataset. Transforms Gaussian noise → two moons distribution using only the four core equations from the paper.

- Pure CNF vs. Flow Matching comparison
- Velocity field learning vs. memorization
- Key finding: Flow Matching generalizes to unseen noise; pure CNF memorizes training pairs

---

## Structure

```
ml-foundations/
├── probability-statistics/
│   └── 01_bayes_theorem.ipynb
├── flow matching/
│   └── ...
└── README.md
```

---

## Philosophy

> "The only thing we can hold is the thing in front of us."

Top-down, project-first. Every notebook starts with a concrete problem and builds the math only as far as needed to solve it.

---
