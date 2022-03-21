from PIL import Image
import os

# splits images through the middle over the x axis and returns two PIL images
def splitImg_var(pathfromroot:str = None, img:Image = None):
    image = None
    if Image is None:
        image = Image.open(pathfromroot)
    else:
        image = img

    fullW, fullH = image.size
    limg = image.crop((0, 0, fullW/2, fullH))
    rimg = image.crop((fullW/2, 0, fullW, fullH))

    return limg, rimg

def splitImgs_dir(pathfromroot, outputpath):
    count = 0

    path = os.getcwd() + pathfromroot
    outputpath = os.getcwd() + outputpath

    for img in os.listdir(path):
        if img.endswith(".png"):
            image = Image.open(path + img)
            fullW, fullH = image.size
            left_image = image.crop((0, 0, fullW/2, fullH))
            right_image = image.crop((fullW/2, 0, fullW, fullH))
            left_image.save(outputpath + '/left' + str(count) + ".png")
            right_image.save(outputpath + '/right' + str(count) + ".png")
            count = count + 1
            print(count)
    print('FINISHED')

