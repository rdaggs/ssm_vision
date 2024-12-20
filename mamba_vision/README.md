# Steps to install
`git clone https://github.com/Dao-AILab/causal-conv1d.git`
`cd causal-conv1d`
`git checkout v1.0.2`
`CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .`

`cd {project_dir}`
`git clone https://github.com/state-spaces/mamba.git`
`cd mamba`
`pip install .`

`cd {project_dir}`
`git clone https://github.com/NVlabs/MambaVision.git`
remove causal-conv1d and mamba-ssm from requirements.txt and setup.py
`cd MambaVision`
`pip install .`
