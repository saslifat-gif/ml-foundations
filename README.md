# Toy Flow Matching from Scratch

Implementing Flow Matching (Lipman et al. 2022) from scratch on a 2D toy dataset.

## Goal
Transform Gaussian noise → two moons distribution using only the four core equations from the paper.

## Key Finding
- Pure CNF memorizes training pairs (loss→0, fails on new data)
- Flow Matching learns the velocity field (generalizes to unseen noise)
