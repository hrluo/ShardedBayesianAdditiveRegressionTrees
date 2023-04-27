


# ShardedBayesianAdditiveRegressionTrees

**Content**
This is the code repository for the research publication "Sharded Bayesian Additive Regression Trees" (abbreviated SBT) by Hengrui Luo and [Matthew T. Pratola](http://www.matthewpratola.com/). 
The manuscript of this paper can be accessed at https://arxiv.org/abs/2305.xxxx. 

 - In [experiment folder](https://github.com/hrluo/ShardedBayesianAdditiveRegressionTrees/tree/master/experiment), we provided a the R and bash code that reproduce the BART and SBT Branin/Friedman synthetic datasetsfor large datasets. The root folder contains the [openBT](https://bitbucket.org/mpratola/openbt/src/master/) software distribution we used to conduct the experiments, including compilable C++ source code. Please refer `README_openBT.md` to set up the distribution. Our environment is Ubuntu 22.04. 
 - In [experiment/redshift folder](https://github.com/hrluo/ShardedBayesianAdditiveRegressionTrees/tree/master/experiment/redshift), we provided the actual redshift dataset we used to test the scalability in our paper. 

**Abstract**
In this paper we develop the randomized Sharded Bayesian Additive Regression Trees (SBT) model.

We introduce a randomization auxiliary variable and a sharding tree to decide partitioning of data, and fit each partition component to a sub-model using Bayesian Additive Regression Tree (BART). By observing that the optimal design of a sharding tree can determine optimal sharding for sub-models on a product space, we introduce an intersection tree structure to completely specify both the sharding and modeling using only tree structures. In addition to experiments, we also derive the theoretical optimal weights for minimizing posterior contractions and prove the worst-case complexity of SBT. 

**Citation**
We provided R production code for reproducible and experimental purposes under [LICENSE](https://github.com/hrluo/ShardedBayesianAdditiveRegressionTrees/blob/master/LICENSE).
Please cite our paper using following BibTeX item:

    @article{luopratola2023shard,
        title={Sparse Additive Gaussian Process Regression},
        author={Hengrui Luo and Matthew T. Pratola},
        year={2023},
        eprint={2305.xxxx},
        archivePrefix={arXiv},
        primaryClass={math.ST}
    }

Thank you again for the interest and please reach out if you have further questions.
