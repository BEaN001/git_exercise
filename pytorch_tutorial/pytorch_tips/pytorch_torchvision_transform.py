from torchvision import transforms
from PIL import Image
crop = transforms.Scale(12)
img = Image.open('/Users/binyu/Documents/git_exercise/computer_vision/datas/mouth/0/1neutral.jpg')

print(type(img))
print(img.size)

croped_img = crop(img)
print(type(croped_img))
print(croped_img.size)

