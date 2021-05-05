1)
SLAM_manual_control.py is the manual control SLAM version

2)
main.py inside pytorch-a2c-ppo-acktr (or pytorch_SLAM_main.py) is the RL+SLAM version

Usage:
python3 main.py --algo ppo --num-frames 5000000 --num-processes 16 --num-steps 80 --lr 0.00005 --env-name MiniWorld-Hallway-v0


slam_settings.yaml is SLAM setting file, vocabulary file is default file in orb-slam2. It's too big to put here