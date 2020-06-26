# Conflict Resolution Module
An implementation of a `conflictFinder` module based on the publication of Olivier et al. [1]
The main idea is to provide the python-mip library with a conflict dealing module

## Instalation 

`pip install mip>=1.9.0`
Currently this is a separated module from the library, so you just need to import the `conflict.py`to your code

## Usage
This module implements two classes `ConflictFinder` and the `ConflictResolver` class. the first one is an implementation of a few IIS finder algorithms and the second one is the implementation of a relaxation algorithm. 

### The `ConflictFinder` class (The IIS)

#### tldr 

```
cf = ConflictFinder()
iis = cf.find_iis(model = my_infeasible_model, method='deletion-filter') # set of infeasible constraints
```
####  long explanation
An IIS stands for Irreducible Infeasible Set of constraints. on a infeasible model you can have one or many infeasible sets, and some of them can be linked between them. Let's for example define a infeasible linear model with 4 constraints:

* `c1: x>=3`
* `c2: x<=1`
* `c3: y>=3`
* `c4: y<=1`

we can see that there are 2 IIS on the upper set  `IIS_1 = [c1,c2], IIS_2 = [c3,c4]`. This case is evident to see, that we only have two sets, but lets add a fifth constraint

* `c5: y>=4`

now we have a third IIS  `IIS_3 = [c4,c5]`, we can realized that the problem of finding all the infeasibilities needs to search all the combinations. And that is a hard problem, usually you just need to apply some relaxation algorithm once you are debugging so this class will only provide you a way to find one IIS. 

currently there are two methods implemented, `'deletion-filter'` and `'additive_algorithm'` **this two methods only work for linear infeasibilities**. mip infeasibilities (when the feasible region does not contain integer solutions) **are not supported yet**.


### The `ConflictResolver` class (The hierarchy relaxation algorithm)

#### tldr 
```
    # all the constraints have a `_l{i}` in the crt.name where i is the level of importance i in [1, ... , 7] 
    # where 1 is the lowest level, and 7 is the mandatory level, that means that is never to be relaxed

    # resolve a conflict
    cr = ConflictResolver()
    relaxed_model = cr.hierarchy_relaxer(infeasible_model, relaxer_objective = 'min_abs_slack_val' )
```
####  long explanation


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
 
