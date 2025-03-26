import os
import socket
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
import sqlite3
from PIL import Image

from src.DataLoader import CaltechBirdsDataset

if socket.gethostname() == 'LTSSL-sKTPpP5Xl':
    data_dir = 'C:\\Users\\ams90\\PycharmProjects\\ConceptsBirds\\data'
elif socket.gethostname() == 'LAPTOP-NA88OLS1':
    data_dir = 'D:\\data\\caltecBirds\\CUB_200_2011'
elif socket.gethostname() == 'andrew-ubuntu':
    data_dir = '/home/andrew/Data/CUB_200_2011'
else:
    data_dir = '/home/bwc/ams90/datasets/caltecBirds/CUB_200_2011'



class ImageViewer:
    def __init__(self, data_dir):

        self.dataloader = CaltechBirdsDataset(bounding=False)

        self.conn = sqlite3.connect(database=os.path.join(data_dir, 'birds.db'))
        cursor = self.conn.cursor()
        self.class_dict = {class_id: class_name for (class_id, class_name) in cursor.execute("select class_id, class_name from classes").fetchall()}

        cursor.close()

        self.class_id = 1
        self.images_list = self.get_indexes(self.class_id)
        self.index = 0

        # Create the initial figure and axis
        self.fig, self.ax = plt.subplot_mosaic("AA;BC", height_ratios=[1,8], figsize=(16, 8))

        data_dict, image = self.dataloader.getitem(self.index)
        self.display_menu()
        self.display_image(image)
        self.display_text(data_dict['file_name'])

        # Connect event handlers for key press
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Display the plot
        plt.show()

    def get_indexes(self, class_id):
        self.conn = sqlite3.connect(database=os.path.join(data_dir, 'birds.db'))
        cursor = self.conn.cursor()
        images_list = [image_id for (image_id, filename) in
                           cursor.execute("select image_id, filename from images where class_id = ?", (class_id,)).fetchall()]
        print(len(images_list), images_list)
        return images_list


    def display_menu(self):

        self.ax['A'].clear()
        self.ax['A'].axis('off')
        self.ax['A'].text(0.5,0.5,self.class_dict[self.class_id], ha='center', family='monospace',  fontsize=12)



    def display_text(self, file_name):

        # Clear the axes and remove numbers/ticks
        self.ax['C'].clear()
        self.ax['C'].axis('off')
        self.ax['C'].set_title(file_name, family='monospace',  fontsize=10)

        image_cursor = self.conn.cursor()
        (image_id, ) = image_cursor.execute("select image_id from images where filename = ?", (file_name,)).fetchone()

        text = ""

        concept_cursor = self.conn.cursor()
        attribute_cursor = self.conn.cursor()

        for (concept_id, concept_name) in concept_cursor.execute("""select   concept_id,
                                                                             concept_name
                                                                    from     concepts
                                                                    order by concept_name""").fetchall():
            text += concept_name.ljust(21) + ': '

            for (value, certainty) in attribute_cursor.execute("""select  a.value,
                                                                              ia.certainty
                                                                      from    attributes a,
                                                                              image_attributes ia
                                                                      where   ia.image_id    = ?
                                                                      and     a.attribute_id = ia.attribute_id
                                                                      and     a.concept_id   = ?
                                                                      and     ia.present     = 1
                                                                      order by a.value""", (image_id, concept_id)).fetchall():
                text += (value + ' ' + str(certainty)).ljust(23)


            text += '\n'



        self.ax['C'].text(0, 0.9, text,  va='top', family='monospace', linespacing=1.7, fontsize=8)

    def display_image(self, image):
        """Displays the current image with its associated data."""

        # Clear the axes and remove numbers/ticks
        self.ax['B'].clear()
        self.ax['B'].axis('off')
        self.ax['B'].imshow(image)



    def on_key_press(self, event: KeyEvent):
        """Handles key press events to navigate through the images."""
        if event.key == 'right':  # Right arrow key
            self.index = self.index + 1 # Go to next image
            if self.index == len(self.images_list): self.index = 0


        elif event.key == 'left':  # Left arrow key
            self.index = self.index - 1  # Go to previous image
            if self.index == - 1: self.index = len(self.images_list) - 1

        elif event.key == 'up':  # Up arrow key
            self.class_id = self.class_id - 1  # Go to next class
            if self.class_id == 0: self.class_id = 200
            self.images_list = self.get_indexes(self.class_id)
            self.index = 0

        elif event.key == 'down':  # Down arrow key
            self.class_id = self.class_id + 1  # Go to previous class
            if self.class_id == 201: self.class_id = 1
            self.images_list = self.get_indexes(self.class_id)
            self.index = 0


        # Update the display with the new image and data
        data_dict, image = self.dataloader.getitem(self.images_list[self.index]-1)

        self.display_menu()
        self.display_image(image)
        self.display_text(data_dict['file_name'])

        # Redraw the canvas
        self.fig.canvas.draw()


# Create and run the viewer
viewer = ImageViewer(data_dir=data_dir)
