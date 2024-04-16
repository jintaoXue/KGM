# KGM
knowledge graph mapping 


# Ubuntu setup

conda create env -f KGM_conda_env_simple.yml
then ref https://www.dgl.ai/pages/start.html to install dgl 

# Windows
conda create --name KGM python=3.10
then ref https://www.dgl.ai/pages/start.html to install dgl like:
    conda install -c dglteam/label/cu118 dgl
conda env update -n KGM --file KGM_conda_windows.yml
install torch: https://download.pytorch.org/whl/torch_stable.html
    torch2.0.1 cu118
    https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-win_amd64.whl
    https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp311-cp311-win_amd64.whl
