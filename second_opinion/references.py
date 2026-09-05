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
    "hastie1986": (
        "Hastie, T., & Tibshirani, R. (1986). Generalized additive models. "
        "Statistical Science, 1(3), 297-318. doi:10.1214/ss/1177013604"
    ),
    "angelopoulos2021": (
        # Two versions exist and they are easy to conflate. The 2021 arXiv
        # preprint is "A Gentle Introduction to Conformal Prediction and
        # Distribution-Free Uncertainty Quantification"; the journal version is
        # "Conformal Prediction: A Gentle Introduction", Foundations and Trends
        # in Machine Learning 16(4), 494-591, in 2023. An earlier draft of this
        # file cited the preprint's year and title against a volume/page range
        # that belongs to neither ("4(1-2), 1-210"). The journal version is
        # cited here; the preprint arXiv ID is kept because it is the copy most
        # readers will actually open.
        "Angelopoulos, A.N., & Bates, S. (2023). Conformal prediction: a gentle "
        "introduction. Foundations and Trends in Machine Learning, 16(4), "
        "494-591. doi:10.1561/2200000101 "
        "(preprint: arXiv:2107.07511, 2021)"
    ),
}

# Ordered for display in README, UI, and evaluation output.
REFERENCES_BIBLIOGRAPHY: list[str] = [
    REFERENCES["street1993"],
    REFERENCES["guo2017"],
    REFERENCES["lee2018"],
    REFERENCES["sadinle2019"],
    REFERENCES["hastie1986"],
    REFERENCES["angelopoulos2021"],
]
