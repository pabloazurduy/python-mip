# Conflict Resolution Module
An implementation of a `conflictFinder` module based on the publication of Olivier et al. [1]
The main idea is to provide the python-mip library with a conflict dealing module

## Instalation 

`pip install mip>=1.9.0`
Currently this is a separated module from the library, so you just need to import the `conflict.py`to your code

## Usage
This module implements two classes `ConflictFinder` and the `ConflictResolver` class. the first one is an implementation of a few IIS finder algorithms and the second one is the implementation of a relaxation algorithm. 

### The `ConflictFinder` class (The IIS)


### The `ConflictResolver` class (The hierarchy relaxation algorithm)



### TODO
#### IIS algorithms 
- [x] Implement Deletion Filter Algorithm (LP)
- [x] Implement Additive Algorithm (LP) #bug

 <img src="img/MILP_infeasibility.png" alt="alt text" width="200"/>

- [ ] Implement Deletion Filter Algorithm (IR-LC-BD) (MIPLP)
- [ ] Implement Deletion Filter Algorithm (LC-IR-BD) (MIPLP)

#### Relaxation module 
- [ ] Implement a linear punishment relaxation algorithm (based on a hierarchy structure)


#### References 
[1] [OLIVIER GUIEU AND JOHN W. CHINNECK 1998](http://www.sce.carleton.ca/faculty/chinneck/docs/GuieuChinneck.pdf)
 
