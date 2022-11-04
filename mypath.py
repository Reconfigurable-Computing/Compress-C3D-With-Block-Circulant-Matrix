class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/Path/to/UCF-101'

            # Save preprocess data into output_dir
            #output_dir = 'D:/UCF101_DATASET/UCF101_DATASET/ucf101'
            output_dir = '/mnt/data/UCF101_DATASET/ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '../hmdb51'

            output_dir = '../VAR/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/path/to/Models/c3d-pretrained.pth'