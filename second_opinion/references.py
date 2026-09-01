"""Bibliography for methods used in Second Opinion.

Every load-bearing algorithm in this project traces to a published source.
Centralised here so README, UI, and evaluation scripts cite the same strings.
"""

from __future__ import annotations

REFERENCES: dict[str, str] = {
    "street1993": (
        "Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear "
        "feature extraction for breast tumor diagnosis. IS&T/SPIE International "
        "Symposium on Electronic Imaging: Science and Technology, 1905, 861-870. "
        "doi:10.1117/12.148698"
    ),
    "guo2017": (
        "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On "
        "calibration of modern neural networks. Proceedings of the 34th "
        "International Conference on Machine Learning (ICML), 70, 1321-1330. "
        "arXiv:1706.04599"
    ),
    "lee2018": (
        "Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified "
        "framework for detecting out-of-distribution samples. Advances in "
        "Neural Information Processing Systems (NeurIPS), 31. "
        "arXiv:1807.03888"
    ),
    "sadinle2019": (
        "Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous "
        "set-valued classifiers with bounded error levels. Journal of the "
        "American Statistical Association, 114(525), 223-234. "
        "doi:10.1080/01621459.2017.1395341"
    ),
    "angelopoulos2021": (
        "Angelopoulos, A.N., & Bates, S. (2021). A gentle introduction to "
        "conformal prediction and distribution-free uncertainty quantification. "
        "Foundations and Trends in Machine Learning, 4(1-2), 1-210. "
        "arXiv:2107.07511"
    ),
}

# Ordered for display in README, UI, and evaluation output.
REFERENCES_BIBLIOGRAPHY: list[str] = [
    REFERENCES["street1993"],
    REFERENCES["guo2017"],
    REFERENCES["lee2018"],
    REFERENCES["sadinle2019"],
    REFERENCES["angelopoulos2021"],
]
