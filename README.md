# hsbc-synthetic-code
Scripts from Turing DSG for generating synthetic transaction data. Working group:

* Alex Bird (University of Edingburgh)
* Alexandre Navarro (University Of Cambridge)
* Ayman Boustati (University of Warwick)
* Chen Zhang (University College London)
* Nicolai Baldin (University of Cambridge)
* Nitin Agrawal (University of Oxford)
* Saranya Govindan (Imperial College London, Tesco Plc)
* Varuna de Silva (Loughborough University)

## Brief Overview
The project focussed on ideas rather than a working implementation or proof-of-concept. While considerable effort was made to use open datasets, the conjoining of the government spending data and the czech bank data did not yield meaningful relationships (as was observed at the end of the project by using a random forest to predict destinations from covariates). The resulting code is therefore only of partial use. Note also that Nikolai's thoughts regarding VAEs, GANs and Copulas did
not have accompanying code. The (approximation) of the generative model is as follows:

1. *lda_stuff / lda.ipynb* Performing LDA upon customers and payees. Payees ideally require preprocessing such that locations or alternate spellings etc. yield the same string. The output is a clusters of payee types and a distribution over types per customer.
2. *nn_stuff / Neural Net.ipynb* In which the goal was to predict the distribution over types for each customer. While a standard feedforward net can in principal perform well here, it was impossible to tune it since there is no signal in the data. We do not expect this to perform well out of the box - standard techniques of dropout, batch / layer normalisation, hyperparameter and architecture normalisation will be required.
3. *gp_stuff / Transaction Generator Using GPs* In which the goal was to predict transaction amounts in each type per customer.
4. *ABM / ABM_Implementation* In which ideas for agent-based rules were tested.

Some attempts at writing the full model in STAN may be found in the `lda_stuff` folder. However, STAN does not currently handle discrete variables due to the Hamiltonian mechanics involved in its sampling procedure. Automatic Differentiation using Variational Inference is also implemented in this environment. One should expect given various papers in NIPS 2016 that we should see implementations of black box generic (variational) inference that can be applied in the
coming year or two.
