https://stackoverflow.com/questions/70080062/how-to-correctly-use-imagedatagenerator-in-keras

Example:


'''
def get_data(self, path, is_vgg, shuffle, augment):
        " Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        "

        if augment:
            # TODO: Use the arguments of ImageDataGenerator()
            #       to augment the data. Leave the
            #       preprocessing_function argument as is unless
            #       you have written your own custom preprocessing
            #       function (see custom_preprocess_fn()).
            #
            # Documentation for ImageDataGenerator: https://bit.ly/2wN2EmK
            #
            # ============================================================
            '''
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40, # See if there is an issue with these arguments
                width_shift_range=0.2, # Try only adding one at a time
                height_shift_range=0.2, 
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest',
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False, # False
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-06,
                brightness_range=None,
                channel_shift_range=0.0,
                cval=0.0,
                data_format=None,
                validation_split=0.0,
                interpolation_order=1,
                dtype=None,
                preprocessing_function=self.preprocess_fn)
            '''
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=30, # See if there is an issue with these arguments
                width_shift_range=0.2, # Try only adding one at a time
                height_shift_range=0.2, 
                #rescale=1./255,
                #shear_range=0.1,
                zoom_range=0.3,
                horizontal_flip=True,
                vertical_flip=True,
                #fill_mode='nearest',
                preprocessing_function=self.preprocess_fn)

            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size = 224 if is_vgg else hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen
'''