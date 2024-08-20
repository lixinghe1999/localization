import numpy as np
mic_array_seeed = np.c_[              
                    [ -0.03,  0.06, 0.0],
                    [ 0.03,  0.06, 0.0],
                    [ 0.06,  0.0, 0.0],
                    [ 0.03,  -0.06, 0.0],
                    [ -0.03,  -0.06, 0.0],
                    [ -0.06,  0, 0.0], 
                    ]

# M1 (45, 35, 4.2cm) (-45, -35, 4.2cm), (135, -35, 4.2cm), (-135, 35, 4.2cm)
mic_array_starss23 = np.c_[[np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45)), np.sin(np.deg2rad(35))],
                           [np.cos(np.deg2rad(-45)), np.sin(np.deg2rad(-45)), np.sin(np.deg2rad(-35))],
                            [np.cos(np.deg2rad(135)), np.sin(np.deg2rad(135)), np.sin(np.deg2rad(-35))],
                            [np.cos(np.deg2rad(-135)), np.sin(np.deg2rad(-135)), np.sin(np.deg2rad(35))]] * 0.942


mic_array_binaural = np.c_[[0.1, 0, 0], [-0.1, 0, 0.0]]