import os
import numpy
import pandas
import pathlib
from pathlib import Path
from ipywidgets import Output, Button, Layout, VBox, HBox, Image
from IPython.display import display, clear_output
from shutil import copyfile

class LabellingTool():

    def __init__(self, classes: dict, path: str):
        self.classes: list[str] = classes.keys()
        self.class_savepath: dict = classes
        self.imgdir: str = path
        self.image_paths: list[pathlib.WindowsPath] = [f for f in self.imgdir.glob('*.jpg')]
        self.position: int = 0
        self.labels: dict = {}
        self.buttons: HBox = self.setbuttons()
        self.frame: VBox = VBox()

        self.assert_dirs()

    def run_labelling(self):
        self.image = Image(value=open(self.image_paths[self.position], 'rb').read(), format='jpg')
        self.frame = VBox([self.image, self.buttons])
        display(self.frame)


    def assert_dirs(self):
        for dir in self.class_savepath.values():
            if not os.path.exists(dir):
                os.makedirs(dir)

    
    def setbuttons(self):
        class_buttons = []
        for label in self.classes:
            class_button = Button(description=label)
            class_button.on_click(self.save_class)
            class_buttons.append(class_button)
        return HBox(class_buttons, layout=Layout(justify_content='center'))
    
    
    def updateframe(self):
        self.image = Image(value=open(self.image_paths[self.position], 'rb').read(), format='jpg')
        self.frame = VBox([self.image, self.buttons])
        display(self.frame)

    
    def label_data(self):
        """
        Main function to label the data. 
        """
        image = Image(filename=self.image_paths[self.position])
        frame = Output(layout=Layout(height="300px", max_width="300px"))
        with frame:
            display(image)


    def save_class(self, button: Button):
        """
        Save the class of the current image to a dictionary and copy the image to the corresponding folder.
        """
        # copy image to the corresponding folder
        currentpath = self.image_paths[self.position]
        station_id = os.path.splitext(currentpath.name)[0]
        label = button.description
        copyfile(currentpath, os.path.join(self.class_savepath[label], currentpath.name))
        self.labels[station_id] = label

        # move to the next image
        self.next_image()


    def next_image(self):
        """
        Show the next image in the list.
        """
        self.position += 1
        if self.position >= len(self.image_paths):
            print('No more images to label.')
            self.logging()
            return

        self.image = Image(value=open(self.image_paths[self.position], 'rb').read(), format='jpg')
        frame = VBox([self.image, self.buttons])
        clear_output()
        display(frame)

    
    def logging(self):
        """
        Log the labels to a csv file.
        """
        savepath = os.path.join(os.path.dirname(self.imgdir), 'labels.csv')
        df = pandas.DataFrame(self.labels.items(), columns=['station_id', 'label'])
        df.to_csv(savepath, index=False)
        print('Labels saved to labels.csv')
        return df
    