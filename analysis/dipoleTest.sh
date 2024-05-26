#!/bin/bash
#SBATCH --job-name=dipoleTest
#SBATCH -A icei_H_King
#SBATCH --partition=g100_usr_prod
#SBATCH --qos=g100_qos_dbg
#SBATCH -t 1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -o dipoleTest.run
#SBATCH -e dipoleTest.err
#SBATCH --mail-user=erica.griffith@downstate.edu
#SBATCH --mail-type=end

source ~/.bashrc

cd /g100/home/userexternal/egriffit/A1/analysis/

srun python dipoleTest.py

wait