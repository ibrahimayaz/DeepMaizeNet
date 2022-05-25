#!/bin/bash
#SBATCH -N 3
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH -A akademik
#SBATCH --output=result.txt
#SBATCH -e J.err
#SBATCH --job-name=ibrahim-misir


module load shared
module load slurm
module load python
#module load anaconda
module load spack

#echo "misir_son2.py orijinal calistiriliyor..."
#python /home/iayaz/misir/misir_son2.py --base "orijinal" --channel 3 --epoch 200
#mv result.txt orijinal


echo "misir_son2.py orijinal calistiriliyor..."
python /home/iayaz/misir/misir_son2.py --base "orijinal" --channel 3 --epoch 350
mv result.txt orijinal
#echo "misir_son2.py calistiriliyor... channel 1 base ifcm_median3"
#python /home/iayaz/misir/misir_son2.py --base "ifcm_median3" channel 1
