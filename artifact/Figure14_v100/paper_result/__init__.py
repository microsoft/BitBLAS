b1s1_providers = ["End2End LLAMA"]
b1s1_times_data = [
    ("Bitter", [1.0305]),
    ("Bitter-W$_{INT8}$A$_{FP16}$", [0.599533535]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [0.345816122]),
    ("Bitter-W$_{INT2}$A$_{FP16}$", [0.285322012]),
    ("Bitter-W$_{INT1}$A$_{FP16}$", [0.266121405]),
    ("Bitter-W$_{INT8}$A$_{INT8}$", [0.559497633]),
    ("Bitter-W$_{INT4}$A$_{INT8}$", [0.341658231]),
    ("Bitter-W$_{INT2}$A$_{INT8}$", [0.207215499]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [0.161347526]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [0.341658231]),
    ("Bitter-W$_{INT2}$A$_{INT4}$", [0.207215499]),
    ("Bitter-W$_{INT1}$A$_{INT4}$", [0.161347526]),
]

b1s4096_providers = ["End2End LLAMA"]
b1s4096_times_data = [
    ("Bitter", [33.7857]),
    ("Bitter-W$_{INT8}$A$_{FP16}$", [35.63580809]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [32.27541218]),
    ("Bitter-W$_{INT2}$A$_{FP16}$", [35.66067246]),
    ("Bitter-W$_{INT1}$A$_{FP16}$", [34.7067301]),
    ("Bitter-W$_{INT8}$A$_{INT8}$", [27.73186896]),
    ("Bitter-W$_{INT4}$A$_{INT8}$", [24.65174276]),
    ("Bitter-W$_{INT2}$A$_{INT8}$", [24.34116963]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [24.39685316]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [13.99112791]),
    ("Bitter-W$_{INT2}$A$_{INT4}$", [14.26845708]),
    ("Bitter-W$_{INT1}$A$_{INT4}$", [14.27030029]),
]

b1s1_matmul_providers = ["M0", "M1", "M2", "M3"]
b1s1_matmul_times_data = [
    ("Bitter", [0.0086016, 0.080486402, 0.265011191, 0.270745605]),
    (
        "Bitter-W$_{INT8}$A$_{FP16}$",
        [0.007836564, 0.04614396, 0.145203829, 0.149608821],
    ),
    (
        "Bitter-W$_{INT4}$A$_{FP16}$",
        [0.006014286, 0.023184372, 0.077943005, 0.07997679],
    ),
    (
        "Bitter-W$_{INT2}$A$_{FP16}$",
        [0.005287193, 0.017449051, 0.061230421, 0.065832675],
    ),
    (
        "Bitter-W$_{INT1}$A$_{FP16}$",
        [0.005552147, 0.018020362, 0.056302968, 0.054814443],
    ),
    ("Bitter-W$_{INT8}$A$_{INT8}$", [0.00567061, 0.043749936, 0.13650924, 0.136082053]),
    (
        "Bitter-W$_{INT4}$A$_{INT8}$",
        [0.005795642, 0.023643596, 0.076373927, 0.078475893],
    ),
    (
        "Bitter-W$_{INT2}$A$_{INT8}$",
        [0.004338192, 0.009642712, 0.040453974, 0.046789736],
    ),
    (
        "Bitter-W$_{INT1}$A$_{INT8}$",
        [0.004221722, 0.010228677, 0.026232524, 0.028425673],
    ),
    (
        "Bitter-W$_{INT4}$A$_{INT4}$",
        [0.005795642, 0.023643596, 0.076373927, 0.078475893],
    ),
    (
        "Bitter-W$_{INT2}$A$_{INT4}$",
        [0.004338192, 0.009642712, 0.040453974, 0.046789736],
    ),
    (
        "Bitter-W$_{INT1}$A$_{INT4}$",
        [0.004221722, 0.010228677, 0.026232524, 0.028425673],
    ),
]

b1s4096_matmul_providers = ["M0", "M1", "M2", "M3"]
b1s4096_matmul_times_data = [
    ("Bitter", [0.35203889, 2.239631414, 7.323716164, 7.485508442]),
    (
        "Bitter-W$_{INT8}$A$_{FP16}$",
        [0.376824141, 2.289058924, 7.878970623, 8.076682091],
    ),
    (
        "Bitter-W$_{INT4}$A$_{FP16}$",
        [0.356162071, 2.242493153, 6.808439732, 6.991803646],
    ),
    (
        "Bitter-W$_{INT2}$A$_{FP16}$",
        [0.354000181, 2.387015343, 7.863059521, 7.983103752],
    ),
    (
        "Bitter-W$_{INT1}$A$_{FP16}$",
        [0.371727765, 2.249416351, 7.629824162, 7.735374928],
    ),
    (
        "Bitter-W$_{INT8}$A$_{INT8}$",
        [0.262480676, 1.687499881, 5.821496964, 5.719495296],
    ),
    ("Bitter-W$_{INT4}$A$_{INT8}$", [0.219333276, 1.43702817, 4.975811005, 4.91797924]),
    (
        "Bitter-W$_{INT2}$A$_{INT8}$",
        [0.216822594, 1.419834495, 4.880335331, 4.837766171],
    ),
    (
        "Bitter-W$_{INT1}$A$_{INT8}$",
        [0.242987201, 1.525666952, 4.79305172, 4.804022789],
    ),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [0.108979389, 0.624639988, 1.9839499, 2.08657074]),
    ("Bitter-W$_{INT2}$A$_{INT4}$", [0.108979389, 0.624639988, 1.9839499, 2.08657074]),
    ("Bitter-W$_{INT1}$A$_{INT4}$", [0.108979389, 0.624639988, 1.9839499, 2.08657074]),
]