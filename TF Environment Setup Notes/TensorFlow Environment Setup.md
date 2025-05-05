For tensorflow to work with GPU support in 2025:
https://www.youtube.com/watch?v=0S81koZpwPA

First install miniconda in WSL, then conda + pip install everything in WSL:

#shell:
wsl
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
#TO INITIALIZE THE SHELL ENTER:
"yes"

conda create --name tfenv python=3.11
conda activate tfenv

#Make sure you have cuda 12:
#NOTE: on any given system, you can have multiple CUDAs installed, but only one graphics driver
nvidia-smi

pip install --upgrade pip
pip install tensorflow[and-cuda]

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

pip install numpy pandas matplotlib scikit-image
#note: only install one opencv package: https://pypi.org/project/opencv-python-headless/
~~pip install opencv-python opencv-contrib-python~~
 pip install opencv-python-headless
conda install jupyter scikit-learn

now open vs code in wsl conda env:
https://tsdat.readthedocs.io/en/latest/tutorials/setup_wsl/#setting-up-vscode-to-run-with-ubuntu-wsl

open vs code in desired project
#in wsl:
code .
#trust authors
exit new vs code window
#in original vs code instance
ctrl + shift + p
"Reload Window"
ctrl + shift + p
"Reopen Folder in WSL"

NOTE: HAD ISSUE WITH INITIALIZING MINICONDA (DELETED FOLLOWING LINES FROM .bashrc):
https://askubuntu.com/questions/1143142/conda-init-gives-me-no-action-taken
nano ~/.bashrc

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/rs/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/rs/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/rs/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/rs/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

Then just re-install miniconda
~/miniconda3/uninstall.sh --remove-caches --remove-config-files {user,system,all} --remove-user-data
./Miniconda3-latest-Linux-x86_64.sh
