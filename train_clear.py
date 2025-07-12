import sys
import os

sys.path.append(os.path.abspath('./Image-Generator'))

from DCGAN import train

if __name__ == "__main__":
    train(
        dataroot=r"C:\Users\suhan\OneDrive\Documents\Desktop\Internship Project DHI\Image-Generator\datasets",
        num_epochs=20,
        outputD="./Image-Generator/checkpoints/netD.pth",
        outputG="./Image-Generator/checkpoints/netG.pth"
    )
