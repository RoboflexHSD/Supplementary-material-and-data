# Supplementary material and data
Supplementary material and data for publication “Continual learning for neural regression networks to cope with concept drift in industrial processes using convex optimisation”.
#### Abbreviations
Continual Learning Neural Network (CLNN) <br />
Extreme Learning Machine (ELM) <br />
Gridded Ring Buffer (GRB)
# Note
For correct execution of every script contained in this repository, the MATLAB-Software is necessary. Additionally, the following Toolboxes are required:
- MATLAB Deep Learning Toolbox: https://de.mathworks.com/products/deep-learning.html
- MATLAB Optimization Toolbox: https://de.mathworks.com/products/optimization.html
# Relevant Content
### Folders
- Data
- Doc
- Results
### m-Files (Matlab Scripts)
- CLNN.m
- ELM.m
- GRB.m
- ParameterstudyCLNN.m
- ParameterstudyELM.m
# Description
### Data
The "Data" folder contains every concept drift dataset ("A.mat", "B.mat", "C.mat") as well as support points for evaluation ("eval_pointsA.mat", "eval_pointsB.mat", "eval_pointsC.mat") used in publication. Additionally, the initial neural network training dataset ("init.mat"), representing the initial gaussian shape, is deposited. 
### Doc
Folder "Doc" includes a detailed documentation about every class developed for realisation of the CLNN, ELM as well as the GRB. Inputs, Outputs, Parameters, etc. can be viewed therein. 
### Results
Folder "Results" covers every result acquired during benchmarking and training with datasets mentioned in our publication. This foder is subdivided into the following sub-folders:
- "CLNN-A": Results acquired with dataset A and proposed method
- "CLNN-B": Results acquired with dataset B and proposed method
- "CLNN-C": Results acquired with dataset C and proposed method
- "ELM-A": ELM results acquired with dataset A
### CLNN.m
Matlab Script containing the CLNN-class necessary for continual update of the neural network. For more details, see documentation in folder "Doc". 
### ELM.m
Matlab Script containing an implementation of the ELM. Here, for convenient handling, the ELM is implemented as a class. For more details, see documentation in folder "Doc". 
### GRB.m
Matlab Script containing an implementation of the GRB. Here, for convenient handling, the GRB is implemented as a class. The GRB class is necessary for the correct executiuon of the CLNN. For more details, see documentation in folder "Doc". 
### ParameterstudyCLNN.m
Matlab Script including the carried out parameter study in publication. Here, the use of the class "CLNN" can be viewed exemplarily.
### ParameterstudyELM.m
Matlab Script including the carried out parameter study in publication. Here, the use of the class "ELM" can be viewed exemplarily.
# Citation

    @article{GROTERAMM2023105927,
        title = {Continual learning for neural regression networks to cope with concept drift in industrial processes using convex optimisation},
        author = {Wolfgang Grote-Ramm and David Lanuschny and Finn Lorenzen and Marcel {Oliveira Brito} and Felix Schönig},
        journal = {Engineering Applications of Artificial Intelligence},
        volume = {120},
        pages = {105927},
        year = {2023},
        doi = {https://doi.org/10.1016/j.engappai.2023.105927},
        }
