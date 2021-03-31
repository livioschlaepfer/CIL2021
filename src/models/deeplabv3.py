from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DeepLabv3RunnerClass:
    def __init__(self, config):
        self.config = config

        self.init_model()
        self.init_criterion()

    def init_model(self):
        print('Initializing model')

        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, aux_loss=None)
        
        # Update number of segmentation classes in classifier and auxillary classifier
        model.classifier = DeepLabHead(2048, self.config.num_classes)
        model.aux_classifier = FCNHead(1024, self.config.num_classes)
        
        # Print the model we just instantiated
        print(model)

        self.model = model

    def init_criterion(self):
        # Setup the loss
        # self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss()
    
    def forward(self, inputs):
        outputs = self.model(inputs)['out']
        return outputs

    def loss(self, outputs, labels):
        loss = self.criterion(outputs.float(), labels.float())
        return loss

    def convert_to_png(self, outputs, path):
        # Get sample image shape
        input_image = Image.open(self.config.paths.train_image_dir + '/satImage_001.png')

        for output in outputs:
            output_predictions = output
            output_predictions[output > 0.5] = 1
            output_predictions[output <= 0.5] = 0

            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for i in range(2)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")

            # Plot the semantic segmentation predictions per class
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)

            # Store output
            r.save(self.config.paths.test_output_dir + "/" + path[0].split("/")[-1])
            print("Stored output for", path[0].split("/")[-1])
