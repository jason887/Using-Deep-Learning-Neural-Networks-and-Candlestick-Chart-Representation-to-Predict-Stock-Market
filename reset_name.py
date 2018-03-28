import os
dir = '/home/ros/ROS/Stock-Market-Predcition-using-ResNet/dataset/EWT/training/'
for filename in os.listdir(dir):
    if filename is not '':
        reset_name = filename[1:]
        os.rename("{}{}".format(dir, filename), "{}{}".format(dir, reset_name))
