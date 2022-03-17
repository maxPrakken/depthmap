from PIL import Image
import os


#def splitImgs_dir(pathfromroot):
count = 0

path = os.getcwd() + '/splitimgs/'

for img in os.listdir(path):
    if img.endswith(".png"):
        image = Image.open(path + img)
        left_image = image.crop((0,0, 1920, 1080))
        right_image = image.crop((1920,0, 3840, 1080))
        left_image.save('splitted/left' + str(count) + ".png")
        count = count + 1
        right_image.save('splitted/right' + str(count) + ".png")
        count = count + 1
        print(count/2)
print('FINISHED')

