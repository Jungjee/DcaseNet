import numpy as np
import torch
eps = 1e-8

class metric_manager(object):
    def __init__(self, task, save_dir, model, save_best_only=True):
        if(
            'ASC' not in task and 
            'SED' not in task and 
            'TAG' not in task
        ):  raise ValueError('task unrecognized, got:{}'.format(task))        
        self.task = task
        self.save_dir = save_dir
        self.model = model
        self.save_best_only = save_best_only

        if 'SED' in task:
            self.er = 99.0
            self.f1 = 0.0
        if 'ASC' in task:
            self.acc = 0.0
        if 'TAG' in task:
            self.lwlrap = 0.0

    def update_SED(self, epoch, er, f1):
        if self.save_best_only:
            if er <= self.er or f1 >= self.f1: self.save_model('best_SED.pt')
        else:
            self.save_model('{}.pt'.format(epoch))

        if er <= self.er:
            self.er = er
        if f1 >= self.f1:
            self.f1 = f1
        print('Epoch{:2d}: ER(cur/best):{:.5f}/{:.5f} F1:(cur/best):{:.2f}/{:.2f}\n'.format(
            epoch, er, self.er, f1*100, self.f1*100
        ))

        return

    def update_ASC(self, epoch, acc, conf_mat, l_label):
        if self.save_best_only:
            if acc >= self.acc: self.save_model('best_ASC.pt')
        else:
            self.save_model('{}.pt'.format(epoch))

        if acc >= self.acc:
            self.acc =acc 
        print('Epoch{:2d}: acc:(cur/best):{:.2f}/{:.2f}\n'.format(
            epoch, acc, self.acc
        ))
        return

    def update_TAG(self, epoch, lwlrap):
        if self.save_best_only:
            if lwlrap >= self.lwlrap: self.save_model('best_lwlrap.pt')
        else:
            self.save_model('{}.pt'.format(epoch))

        if lwlrap >= self.lwlrap:
            self.lwlrap =lwlrap 
        print('Epoch{:2d}: lwlrap:(cur/best):{:.2f}/{:.2f}\n'.format(
            epoch, lwlrap, self.lwlrap
        ))
        return

    def save_model(self, name):
        torch.save(self.model.state_dict(), self.save_dir+name)
        return

#####
# Metric: TAG
#####
# All-in-one calculation of per-class lwlrap.
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class
        

#####
# Metric: SED
#####
def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score

def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)

    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER

def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return f1_overall_framewise(O_block, T_block)

def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / (block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return er_overall_framewise(O_block, T_block)

def compute_sed_scores(pred, gt, nb_frames_1s):
    """
    Computes SED metrics for one second segments

    :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param nb_frames_1s: integer, number of frames in one second
    :return:
    """
    f1o = f1_overall_1sec(pred, gt, nb_frames_1s)
    ero = er_overall_1sec(pred, gt, nb_frames_1s)
    scores = [ero, f1o]
    return scores


#####
# Functions for format conversions
#####
def output_format_dict_to_classification_labels(_output_dict, _feat_cls):
    _unique_classes = _feat_cls.get_classes()
    _nb_classes = len(_unique_classes)
    _azi_list, _ele_list = _feat_cls.get_azi_ele_list()
    _max_frames = _feat_cls.get_nb_frames()
    _labels = np.zeros((_max_frames, _nb_classes, len(_azi_list) * len(_ele_list)))

    for _frame_cnt in _output_dict.keys():
        if _frame_cnt < _max_frames:
            for _tmp_doa in _output_dict[_frame_cnt]:
                # Making sure the doa's are within the limits
                _tmp_doa[1] = np.clip(_tmp_doa[1], _azi_list[0], _azi_list[-1])
                _tmp_doa[2] = np.clip(_tmp_doa[2], _ele_list[0], _ele_list[-1])

                # create label
                _labels[_frame_cnt, _tmp_doa[0], int(_feat_cls.get_list_index(_tmp_doa[1], _tmp_doa[2]))] = 1

    return _labels

def classification_label_format_to_output_format(_feat_cls, _labels):
    """
    Converts the seld labels predicted in classification format to dcase output format.

    :param _feat_cls: feature or data generator class instance
    :param _labels: SED labels matrix [nb_frames, nb_classes, nb_azi*nb_ele]
    :return: _output_dict: returns a dict containing dcase output format
    """
    _output_dict = {}
    for _frame_ind in range(_labels.shape[0]):
        _tmp_class_ind = np.where(_labels[_frame_ind].sum(1))
        if len(_tmp_class_ind[0]):
            _output_dict[_frame_ind] = []
            for _tmp_class in _tmp_class_ind[0]:
                _tmp_spatial_ind = np.where(_labels[_frame_ind, _tmp_class])
                for _tmp_spatial in _tmp_spatial_ind[0]:
                    _azi, _ele = _feat_cls.get_matrix_index(_tmp_spatial)
                    _output_dict[_frame_ind].append(
                        [_tmp_class, _azi, _ele])

    return _output_dict

def description_file_to_output_format(_desc_file_dict, _unique_classes, _hop_length_sec):
    """
    Reads description file in csv format. Outputs, the dcase format results in dictionary, and additionally writes it
    to the _output_file

    :param _unique_classes: unique classes dictionary, maps class name to class index
    :param _desc_file_dict: full path of the description file
    :param _hop_length_sec: hop length in seconds

    :return: _output_dict: dcase output in dicitionary format
    """

    _output_dict = {}
    for _ind, _tmp_start_sec in enumerate(_desc_file_dict['start']):
        _tmp_class = _unique_classes[_desc_file_dict['class'][_ind]]
        _tmp_azi = _desc_file_dict['azi'][_ind]
        _tmp_ele = _desc_file_dict['ele'][_ind]
        _tmp_end_sec = _desc_file_dict['end'][_ind]

        _start_frame = int(_tmp_start_sec / _hop_length_sec)
        _end_frame = int(_tmp_end_sec / _hop_length_sec)
        for _frame_ind in range(_start_frame, _end_frame + 1):
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            _output_dict[_frame_ind].append([_tmp_class, _tmp_azi, _tmp_ele])

    return _output_dict

def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), int(_words[3])])
    _fid.close()
    return _output_dict

def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
    _fid.close()

def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits
