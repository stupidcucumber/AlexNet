import tensorflow as tf
import os
import xml.etree.cElementTree as ET
import numpy as np
import cv2

class DatasetLoader():
    """
        Class helps create TensorFlow Dataset from generators.

        To use instantiate DatasetLoader and call it with appropriate image_size and shuffle argument. 
    """
    def __init__(self, root_folder: str=None, volume: str=None):
        """
            root_folder: absolute/relative path to the dataset folder. For now only ImageNet dataset is supported.
            image_size: a tuple like (640, 640) containing the object to classificate.
            shuffle: whether to shuffle the dataset
            volume: which part of the dataset to include ("train"/"val")
        """
        self.root_folder = root_folder

        self.image_folder = os.path.join(root_folder, 'Data/CLS-LOC', volume)
        self.annotations_folder = os.path.join(root_folder, 'Annotations/CLS-LOC', volume)

        self.subset_file_definition = os.path.join(root_folder, 
                                                   'ImageSets/CLS-LOC', 
                                                   'train_cls.txt' if volume == 'train' else 'val.txt')
    

    def __load_metadata_mapping(self, path: str=""):
        """
            Function loads mapping file, where name of the folder is alias for class_id.
            class_id corresponds to the position of the current line in the file.
        """
        mapping = dict()

        with open(path) as f:
            for index, line in enumerate(f.readlines()):
                name, description = line[:-1].split(' ', maxsplit=1)

                mapping[name] = {
                    "description": description,
                    "class_id": index
                }
        return mapping
    

    def __load_metadata_filenames(self):
        file_list = []

        with open(self.subset_file_definition) as f:
            for line in f.readlines():
                filename = line.split(' ')[0]

                image_path = os.path.join(self.image_folder, filename + '.JPEG')
                annotation_path = os.path.join(self.annotations_folder, filename + '.xml')

                if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                    continue

                file_list.append(filename)

        if self.shuffle:
            np.random.shuffle(file_list)

        return file_list
    
    
    def __parse_xml(self, path: str=""):
        """
            Function parses the xml file with the specified path, and returns a list of 
        [('class_name', [xmin, ymin, xmax, ymax]), ...]

            For now function return only 1 bbox per image.
        """
        data = ET.parse(path)

        root = data.getroot()

        coords_tag_names = ['xmin', 'ymin', 'xmax', 'ymax']

        class_names = []
        coords = []

        for name in root.iter('name'):
            class_name = name.text
            class_names.append(class_name)
            break

        for coord_name in coords_tag_names:
            for coord in root.iter(coord_name):
                coords.append(int(coord.text))
                break

        return class_names[0], coords
    

    def __load_objects(self, max: int=0):
        filenames = self.__load_metadata_filenames()
        
        counter = 0
        for filename in filenames:
            image_path = os.path.join(self.image_folder, filename + '.JPEG')
            annotation_path = os.path.join(self.annotations_folder, filename + '.xml')

            annotation = self.__parse_xml(annotation_path)
            counter += 1

            if counter > max and max != 0:
                break
            
            yield image_path, annotation


    def load_meta(self, max_objects: int=0):
        mapping_path = os.path.join(self.root_folder, 'LOC_synset_mapping.txt')
        self.mapping = self.__load_metadata_mapping(path=mapping_path)
        self.objects = self.__load_objects(max=max_objects)


    def generator(self, image_size: tuple=None, shuffle: bool=False, max: int=0):
        self.shuffle = shuffle
        self.image_size = image_size
        
        # Loads mapping, filelist and performs extraction of later used coordinates of bboxes and their classes.
        self.load_meta(max_objects=max)

        # Generating dataset
        for object in self.objects:
            image_path = object[0]
            class_name, coords = object[1]
            try:
                base_image = cv2.imread(image_path)
                x, y, xmax, ymax = coords

                image = base_image[y: ymax, x: xmax]
                image = cv2.resize(image, dsize=self.image_size) / 255.
            except Exception as e:
                tf.print("Caught an exception: ", e)
                tf.print("Image path: ", image_path)
                continue

            annotation = np.zeros(1000)
            annotation[self.mapping[class_name]['class_id']] = 1
            annotation = tf.convert_to_tensor(annotation, dtype=tf.float32)

            yield image, annotation

    
    def __call__(self, image_size: tuple=None, shuffle: bool=False, max: int=0, batch_size: int=1):
        dataset = tf.data.Dataset.from_generator(
            generator=self.generator,
            args=(image_size, shuffle, max),
            output_signature=(
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(1000), dtype=tf.float32)
            )
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=3)

        dataset = dataset.batch(batch_size)

        return dataset