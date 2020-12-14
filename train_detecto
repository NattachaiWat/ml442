import os
import torch
from torchvision import transforms, utils
from detecto.core import Model, Dataset,DataLoader
from detecto import utils, visualize
from detecto.utils import default_transforms

# defaults = default_transforms()
# print(defaults)


training_label_folder = '/home/ubuntu/ton/namecard/MyNameCard/train2/Annotations'
training_image_folder = '/home/ubuntu/ton/namecard/MyNameCard/train2/JPEGImages'
# Images and XML files in separate folders

params_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]),
                                        transforms.Resize(256)
                                        ])
training_dataset = Dataset(training_label_folder, training_image_folder, transform=params_transforms)

testing_label_folder = '/home/ubuntu/ton/namecard/MyNameCard/test/Annotations'
testing_image_folder = '/home/ubuntu/ton/namecard/MyNameCard/test/JPEGImages'
# Images and XML files in separate folders
testing_dataset = Dataset(testing_label_folder, testing_image_folder,transform=params_transforms)


print(training_dataset)
# Print Dataset
# image, target = training_dataset[0]
# print(image, target)

# # Print Dataset
# image, target = testing_dataset[0]
# print(image, target)

# labels
labels = ["name","email","telephone_number","company"]
if not os.path.exists('model_weights.pth'):
    model = Model(labels,
                  #  device=torch.device('cpu')
                )
else:
    model = Model.load('model_weights.pth', labels)


loader = DataLoader(training_dataset, batch_size=2, shuffle=True)
losses = model.fit(loader,testing_dataset, epochs=2000, learning_rate=0.001,
                   gamma=0.2, lr_step_size=50, verbose=True)


#

test_filename ='/home/ubuntu/ton/namecard/MyNameCard/raw_images/test_01.jpg'
image = utils.read_image(test_filename)  # Helper function to read in images
image = params_transforms(image)

#labels, boxes, scores = model.predict(image)  # Get all predictions on an image
predictions = model.predict_top(image)  # Same as above, but returns only the top predictions

# print(labels, boxes, scores)
print(predictions)

#visualize.show_labeled_image(image, boxes, labels)  # Plot predictions on a single image


# print('training loss: {}'.format(losses))
model.save('model_weights.pth')
