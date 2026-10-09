| model | 0.5 s ahead | 2.5 s ahead | 5 s ahead |
|---|---|---|---|
| no regularizer | 0.35 | 0.12 | 0.14 |
| straightening | 0.38 | 0.12 | 0.11 |
| straightening + pacing | 0.38 | 0.12 | 0.12 |
| random play | 0.44 | 0.18 | 0.19 |
| branch rollouts | 0.39 | 0.12 | 0.11 |
| my demos only | 1.28 | 0.74 | 0.70 |

Latent prediction error divided by the error of assuming nothing moves (lower is better, below 1 beats standing still), over 24 windows from the held-out demos.
