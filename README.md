This is the source code for the paper :

## Joint-Inference-of-selection-and-number-of-selected-target

- **input**: Contains the input haplotype dataset.
- **Distance.py**: Contains the distance metric adaptive-l1-penalized logistics classification [3]. 
- **Inference_s.py**: Contains the main ABC algorithm with specify priors [1][5] 
- **Statistics_new.py**: Contains the summary statistics calculations [4]. 
- **model_mimiCREE.py**: Contains simulation model by using mimiCREE2 [2]. 
- **model_sim.py**: Contains the simulation code for haploid individual using multinomial distribution.

ABCpy pacakage available at [here](https://github.com/eth-cscs/abcpy)
mimiCREE2 user manual available at [here]([https://github.com/eth-cscs/abcpy](https://sourceforge.net/p/mimicree2/wiki/Home/))

[1] Carlo Albert, Hans R Künsch, and Andreas Scheidegger. A simulated annealing approach to approximate bayes computations. Statistics and computing, 25(6):1217–1232, 2015.

[2] Christos Vlachos and Robert Kofler. Mimicree2: Genome-wide forward simulations of evolve and resequencing studies. PLoS computational biology, 14(8):e1006413, 2018.

[3] Hui Zou. The adaptive lasso and its oracle properties. Journal of the American statistical association, 101(476):1418–1429, 2006. 15

[4] Thomas Taus, Andreas Futschik, and Christian Schlötterer. Quantifying selection with pool-seq time series data. Molecular biology and evolution, 34(11):3023–3034, 2017.

[5] Ritabrata Dutta, Marcel Schoengens, Lorenzo Pacchiardi, Avinash Ummadisingu, Nicole Widmer, Pierre Künzli,
Jukka-Pekka Onnela, and Antonietta Mira. Abcpy: A high-performance computing perspective to approximate
bayesian computation. Journal of Statistical Software, 100(7):1–38, 2021.
