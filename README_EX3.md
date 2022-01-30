## Project Setup

This file contains a little bit of information concerning the environment setup of the project. 

The Somoclu library may lead to some problems if the environment contains any versions of any library that do not exactly fit the versions required by Somoclu.

One solution for this is to create an empty Anaconda environment to start with. Then we are able to add the Somoclu library, which will install all the required
packages automatically. After that all the additional packages needed can be installed without a problem.

### Prerequisites

Anaconda is installed on the system https://www.anaconda.com/

### Step 1:

Create an empty environment using the following command:

```conda create --name som_toolbox --no-default-packages```

Make sure that python3 is included in the environment by doing the following:

Activate the environment using

```conda activate som_toolbox```

Type the following command in the terminal to start the python3 interpreter installed in the environment

```python3```


### Step 2:

Follow the tutorial and add conda forge as a channel: 

https://conda-forge.org/docs/user/introduction.html

### Step 3:

Now we can install Somoclu from the conda forge channel using

```conda install somoclu```

This installs all the required packages, so that there won't be an error later on.

### Step 4:

Install all the other required packages
