#!/bin/bash
cmd="$(cat <<eof
apt install tmux vim zsh -y
wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh
sh install.sh
rm install.sh
sed 's/robbyrussell/crcandy/g' -i ~/.zshrc
cd /mmdetection || exit 1
/opt/conda/bin/pip install -e .
/opt/conda/bin/pip install future tensorboard
/opt/conda/bin/python3 -m wandb login e3bfa9973084ce2f7c2169ec76d8af2d1dfcd404
eof
)"

ssh apu_docker_ex "$cmd"