#!/bin/bash
#SBATCH -N 1                        	# number of compute nodes
#SBATCH -n 1                     		# number of tasks your job will spawn
#SBATCH --mem=40G                    	# amount of RAM requested in GiB 
#SBATCH -p htcgpu                       # Run the job on HTC or if want to use GPU use -p gpu (for wall time of > 4 hrs)
#SBATCH -q normal                 	    # Run job under wildfire QOS queue  # For wildfire use -q wildfire
#SBATCH --gres=gpu:A100:1          	    # Request 2 GPU (V100_32, A100) [A100 40GB]
#SBATCH -t 0-4:00                  	    # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             	# STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             	# STDERR (%j = JobId)
#SBATCH --mail-type=ALL             	# Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=agilanka@asu.edu 	# send-to address (Please change to the email user address)

module purge

module avail anaconda

module load anaconda/py3

source activate denoiseHDR

python denoise.py --data /home/agilanka/denoiser/010402_512px_800fr.tif  # Enter the directory in which the codes and the movie are. 

