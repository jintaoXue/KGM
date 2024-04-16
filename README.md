# KGM
knowledge graph mapping 


# Ubuntu setup

conda create env -f KGM_conda_env_simple.yml
then ref https://www.dgl.ai/pages/start.html to install dgl 

# Windows

conda create env -f KGM_conda_windows.yml

install torch: https://download.pytorch.org/whl/torch_stable.html
    torch2.0.1 cu118
    https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-win_amd64.whl

then ref https://www.dgl.ai/pages/start.html to install dgl like:
    pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html 
    conda install -c dglteam/label/cu118 dgl

