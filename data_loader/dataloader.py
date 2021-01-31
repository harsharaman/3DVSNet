from abc import ABCMeta, abstractmethod
#from nnunet.paths import preprocessing_output_dir
import os
import pickle
import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import SpatialTransform, MirrorTransform, GammaTransform, Compose, DataChannelSelectionTransform, SegChannelSelectionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform,     MaskTransform, ConvertSegmentationToRegionsTransform, RemoveKeyTransform
from batchgenerators.augmentations.utils import random_crop_2D_image_batched, pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict

#Implemented from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/dataloading/dataset_loading.py

class SlimDataLoaderBase(object):
    def __init__(self, data, batch_size=2, number_of_threads_in_multithreaded=None):
        """
        Slim version of DataLoaderBase (which is now deprecated). Only provides very simple functionality.
        You must derive from this class to implement your own DataLoader. You must overrive self.generate_train_batch()
        If you use our MultiThreadedAugmenter you will need to also set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!
        :param data: will be stored in self._data. You can use it to generate your batches in self.generate_train_batch()
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        """
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded #change for multi-threaded augmentation
        self._data = data #dictionary of {'data':_of_shape_(b,c,x,y,z), 'seg':_of_shape_(b,c,x,y,z)}
        self.batch_size = batch_size
        self.thread_id = 0
        
    def set_thread_id(self, thread_id):
        self.thread_id = thread_id
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.generate_train_batch()
    
    def __len__(self):
        return len(self._data)
    
    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass
    
# Sample dataloaders (straight and shuffle)
# Not used in our VSNet. I just keep this as guidelines for design principles.
class DummyDL(SlimDataLoaderBase):
    def __init__(self, data, num_threads_in_mt=8): #data = train_dataset, val_dataset
        super(DummyDL, self).__init__(data, batch_size=2, number_of_threads_in_multithreaded=num_threads_in_mt)
        self._data = data
        self.current_position = 0
        self.was_initialized = False
        
    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[idx]
        else:
            self.reset()
            raise StopIteration

class DummyDLWithShuffle(DummyDL):
    def __init__(self, data, num_threads_in_mt=8):
        super(DummyDLWithShuffle, self).__init__(data, num_threads_in_mt=num_threads_in_mt)
        self._data = data
        self.num_restarted = 0
        self.data_order = np.arange(len(self._data))

    def reset(self):
        super(DummyDLWithShuffle, self).reset()
        rs = np.random.RandomState(self.num_restarted)
        rs.shuffle(self.data_order)
        self.num_restarted = self.num_restarted + 1

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[self.data_order[idx]]
        else:
            self.reset()
            raise StopIteration

#Augmentation parameters are adjusted here.
default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 4, #if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
    "num_cached_per_thread": 1,
}

def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)

def unpack_dataset(folder, threads=8, key="data"):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key] * len(npz_files)))
    p.close()
    p.join()

#Returns liver_xxx, where x={1,2,...,n} number of cases
def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers

def load_dataset(folder, num_cases_properties_loading_threshold=1000):
    # we don't load the actual data but instead return the filename to the numpy file.
    print('loading dataset')
    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers: #looping through all cases
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file']) #attach 'properties' key to dataset dict

    return dataset #This return should be the argument for DataLoader3D

from sklearn.model_selection import KFold, train_test_split
def do_split_custom(dataset, fold="normal", dataset_directory=None):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(dataset.keys())
        elif fold == "normal":
            # Divide training and testing data in 80:20 ratio
            all_keys_sorted = np.sort(list(dataset.keys()))
            tr_keys, val_keys = train_test_split(all_keys_sorted, train_size=0.8, random_state=12345)
        
        else:
            splits_file = join(dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                #self.print_to_log_file("Creating new split...")
                splits = []
                all_keys_sorted = np.sort(list(dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            splits = load_pickle(splits_file)

            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
            else:
                self.print_to_log_file("INFO: Requested fold %d but split file only has %d folds. I am now creating a "
                                       "random 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
        
        tr_keys.sort()
        val_keys.sort()
        dataset_tr = OrderedDict()
        for i in tr_keys:
            dataset_tr[i] = dataset[i]
        dataset_val = OrderedDict()
        for i in val_keys:
            dataset_val[i] = dataset[i]
            
        return dataset_tr, dataset_val

#function to get patch size
def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        print(final_shape)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.5, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None, test=False):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by nnUnet preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, it is advised to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training.
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that you will have to store that permanently. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param stage: ignore this 
        :param oversample_foreground_percent=0.5: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D, self).__init__(data, batch_size, 0)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.current_position = 0
        self.was_initialized = False
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.test = test
    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < 126:#Number of threads * idx = required number of patches in one batch
            self.current_position = idx + 1
            if not self.test:
                selected_keys = np.random.choice(self.list_of_keys, self.batch_size, False, None)
            else:
                selected_keys = self.list_of_keys[:self.batch_size]
            data = np.zeros(self.data_shape, dtype=np.float32)
            seg = np.zeros(self.seg_shape, dtype=np.float32)
            case_properties = []
            for j, i in enumerate(selected_keys):
                # oversampling foreground 
                if self.get_do_oversample(j):
                    force_fg = True
                else:
                    force_fg = False

                if 'properties' in self._data[i].keys():
                    properties = self._data[i]['properties']
                else:
                    properties = load_pickle(self._data[i]['properties_file'])
                case_properties.append(properties)

                # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
                # which is much faster to access
                if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                    case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    case_all_data = np.load(self._data[i]['data_file'])['data']
                
                # The following few lines until seg_from_previous_stage = None is irrelevant to our case as we do not use cascade structure.
                # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
                # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
                # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
                # the last channel of the data
                if self.has_prev_stage:
                    if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                        segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                           mmap_mode=self.memmap_mode)[None]
                    else:
                        segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                    # we theoretically support several possible previsous segmentations from which only one is sampled. But
                    # in practice this feature was never used so it's always only one segmentation
                    seg_key = np.random.choice(segs_from_previous_stage.shape[0], replace=False)
                    seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                    assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]),"seg_from_previous_stage does not match the shape ofcase_all_data: %s vs %s" % (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
                else:
                    seg_from_previous_stage = None

                # Relevant from here

                need_to_pad = self.need_to_pad
                for d in range(3):
                    # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                    # always
                    if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]: #case_all_data.shape[d + 1] becuase we are first dimension denotes GT or data
                        need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

                # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
                # define what the upper and lower bound can be to then sample form them with np.random.randint
                shape = case_all_data.shape[1:]
                lb_x = - need_to_pad[0] // 2
                ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
                lb_y = - need_to_pad[1] // 2
                ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
                lb_z = - need_to_pad[2] // 2
                ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

                # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
                # at least one of the foreground classes in the patch
                if not force_fg:
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
                else:
                    # these values should have been precomputed
                    if 'class_locations' not in properties.keys():
                        raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                    # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                    foreground_classes = np.array(
                        [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                    foreground_classes = foreground_classes[foreground_classes > 0]

                    if len(foreground_classes) == 0 and not self.test:
                        # this only happens if some image does not contain foreground voxels at all
                        selected_class = None
                        voxels_of_that_class = None
                        print('case does not contain any foreground classes', i)
                    elif self.test:
                        selected_class = 2
                        voxels_of_that_class = properties['class_locations'][selected_class]
                    else:
                        selected_class = np.random.choice(foreground_classes, replace=False)

                        voxels_of_that_class = properties['class_locations'][selected_class]

                    if voxels_of_that_class is not None:
                        selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                        if self.test:
                            selected_voxel = voxels_of_that_class[0]
                        # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                        # Make sure it is within the bounds of lb and ub
                        bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                        bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                        bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                    else:
                        # If the image does not contain any foreground classes, we fall back to random cropping
                        bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                        bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                        bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

                bbox_x_ub = bbox_x_lb + self.patch_size[0]
                bbox_y_ub = bbox_y_lb + self.patch_size[1]
                bbox_z_ub = bbox_z_lb + self.patch_size[2]

                # We first crop the data to the region of the
                # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
                # valid_bbox is just the coord that lies within the data cube. It will be padded to match the patch size later
                valid_bbox_x_lb = max(0, bbox_x_lb)
                valid_bbox_x_ub = min(shape[0], bbox_x_ub)
                valid_bbox_y_lb = max(0, bbox_y_lb)
                valid_bbox_y_ub = min(shape[1], bbox_y_ub)
                valid_bbox_z_lb = max(0, bbox_z_lb)
                valid_bbox_z_ub = min(shape[2], bbox_z_ub)

                # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
                # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
                # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
                # remove label -1 in the data augmentation but this way it is less error prone)
                case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                        valid_bbox_y_lb:valid_bbox_y_ub,
                                        valid_bbox_z_lb:valid_bbox_z_ub])
                if seg_from_previous_stage is not None:
                    seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                              valid_bbox_y_lb:valid_bbox_y_ub,
                                              valid_bbox_z_lb:valid_bbox_z_ub]

                data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                                      (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                      (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                      (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                 self.pad_mode, **self.pad_kwargs_data)

                seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                        (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                        (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                        (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': -1})
                if seg_from_previous_stage is not None:
                    seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                                                 (-min(0, bbox_x_lb),
                                                                  max(bbox_x_ub - shape[0], 0)),
                                                                 (-min(0, bbox_y_lb),
                                                                  max(bbox_y_ub - shape[1], 0)),
                                                                 (-min(0, bbox_z_lb),
                                                                  max(bbox_z_ub - shape[2], 0))),
                                       'constant', **{'constant_values': 0})

            return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}
        
        else:
            self.reset()
            raise StopIteration
        
#Augmentation done here
def get_moreDA_augmentation(dataloader_train=None, dataloader_val=None, patch_size=None, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, pin_memory=True, regions=None):
    if dataloader_train is not None:
        assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
        tr_transforms = []
        
        #DataChannelSelectionTransform is never used. Unrelated to our dataset
        if params.get("selected_data_channels") is not None:
            tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        # Unrelated to our dataset
        if params.get("dummy_2D") is not None and params.get("dummy_2D"):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None
        
        """
        All spatial transforms are done here . Rotation, deformation, scaling, cropping. Computational time scales 
        only with patch_size, not with input patch size or type of augmentations used.
        Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
        applied (very fast). Interpolation on the image data will only be done at the very end
        Args:
        patch_size (tuple/list/ndarray of int): Output patch size
        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True
        do_elastic_deform (bool): Whether or not to apply elastic deformation
        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval
        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval
        do_rotation (bool): Whether or not to apply rotation
        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!
        do_scale (bool): Whether or not to apply scaling
        scale (tuple of float): scale range ; scale is randomly sampled from interval
        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates
        border_cval_data: If border_mode_data=constant, what value to use?
        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates
        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates
        border_cval_seg: If border_mode_seg=constant, what value to use?
        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])
        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size
        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
        """
        tr_transforms.append(SpatialTransform(
            patch_size, patch_center_dist_from_border=(patch_size//2), do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
        ))
        """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
        each axis is 0.5
        Args:
        axes (tuple of int): axes along which to mirror
        """
        if params.get("do_mirror"):
            tr_transforms.append(MirrorTransform(params.get("mirror_axes")))
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors)
        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                               p_per_sample=params["p_gamma"]))
        # Unrelated to our dataset
        if params.get("mask_was_used_for_normalization") is not None and params.get("mask_was_used_for_normalization"):
            mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
            tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))
        
        #Any -1 value in labels is converted to 0
        tr_transforms.append(RemoveLabelTransform(-1, 0))
        
        #Used for cascade (3D LowRes in combination with 3D HighRes). Unrelated to our model and never used.
        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
            if params.get("cascade_do_cascade_augmentations") and not None and params.get(
                    "cascade_do_cascade_augmentations"):
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size")))
                tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    key="data",
                    p_per_sample=params.get("cascade_remove_conn_comp_p"),
                    fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                    dont_do_if_covers_more_than_X_percent=params.get("cascade_remove_conn_comp_fill_with_other_class_p")))
        
        #'Seg' in dict key is replaced as 'target'. Only conventions.                             
        tr_transforms.append(RenameTransform('seg', 'target', True))
        #Numpy array converted to tensor
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        
        #Consolidation of augmentations with Compose
        tr_transforms = Compose(tr_transforms)
        print('Augment process complete!')
        #Multi-threaded Augmenter used to speeden the augmentation process
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, max(params.get('num_threads')//2,1),
                                                   params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
        return batchgenerator_train
    
    # Spatial Transforms, Mirroring and Gamma augmentation not done for validation   
    elif dataloader_val is not None:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        if params.get("selected_data_channels") is not None:
            val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
                                     
        val_transforms.append(RenameTransform('seg', 'target', True))
                                     
        if regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
            else:
                val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
                                     
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads')//2,1),
                                                   params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
        return batchgenerator_val

# Below is the same dataloader as above, but with Gaussian augmentation
def get_moreDA_augmentation_gaussian(dataloader_train=None, dataloader_val=None, patch_size=None, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, pin_memory=True, regions=None):
    if dataloader_train is not None:
        print('Test condition passed!')
        assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
        tr_transforms = []

        if params.get("selected_data_channels") is not None:
            tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if params.get("dummy_2D") is not None and params.get("dummy_2D"):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None
            
        tr_transforms.append(SpatialTransform(
            patch_size, patch_center_dist_from_border=(patch_size//2), do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
        ))
        """GaussianNoiseTransform: Adds additive Gaussian Noise
        Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution from this interval
        """
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15))
        """GaussianBlurTransform:
        :param blur_sigma:
        :param data_key:
        :param label_key:
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        tr_transforms.append(GaussianBlurTransform((0.5, 1.5), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))

        if params.get("do_mirror"):
            tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                               p_per_sample=params["p_gamma"]))

        if params.get("mask_was_used_for_normalization") is not None and params.get("mask_was_used_for_normalization"):
            mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
            tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))
        
        tr_transforms.append(RemoveLabelTransform(-1, 0))
        
        #tr_transforms.append(RemoveLabelTransform(2, 5))
  
        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
            if params.get("cascade_do_cascade_augmentations") and not None and params.get(
                    "cascade_do_cascade_augmentations"):
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size")))
                tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    key="data",
                    p_per_sample=params.get("cascade_remove_conn_comp_p"),
                    fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                    dont_do_if_covers_more_than_X_percent=params.get("cascade_remove_conn_comp_fill_with_other_class_p")))
                                     
        tr_transforms.append(RenameTransform('seg', 'target', True))
    
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
      
        tr_transforms = Compose(tr_transforms)
        print('Augment process complete!')
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, max(params.get('num_threads')//2,1),
                                                   params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
        return batchgenerator_train
        
    elif dataloader_val is not None:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        if params.get("selected_data_channels") is not None:
            val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
                                     
        val_transforms.append(RenameTransform('seg', 'target', True))
                                     
        if regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
            else:
                val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
                                     
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads')//2,1),
                                                   params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
        return batchgenerator_val


t = "Task003_Liver"
p = os.path.join("/net/pasnas01/pool1/ramanha/corpus/nnUNet/nnUNet_preprocessed", t)
with open(os.path.join(p, "nnUNetPlansv2.1_plans_3D.pkl"), 'rb') as f:
        plans = pickle.load(f)
