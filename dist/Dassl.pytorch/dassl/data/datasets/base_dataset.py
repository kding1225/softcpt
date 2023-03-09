import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
from collections import OrderedDict
import gdown

from dassl.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath
    
    @impath.setter
    def impath(self, value):
        self._impath = value
    
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, value):
        self._label = value
    
    @property
    def domain(self):
        return self._domain
    
    @domain.setter
    def domain(self, value):
        self._domain = value
    
    @property
    def classname(self):
        return self._classname
    
    @classname.setter
    def classname(self, value):
        self._classname = value
    
    def __repr__(self):
        info = "Datum(impath={}, label={}, domain={}, classname={})".format(
            self.impath, self.label, self.domain, self.classname
        )
        return info
    
    
class DatumMT:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (list[int]): class label.
        domain (int): domain label.
        task (list[int]): task id.
    """
    
    def __init__(self, impath="", label=(), domain=0, task=(), box=None):
        assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._task = task
        self._box = box

    @property
    def impath(self):
        return self._impath
    
    @impath.setter
    def impath(self, value):
        self._impath = value
    
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, value):
        self._label = value
    
    @property
    def domain(self):
        return self._domain
    
    @domain.setter
    def domain(self, value):
        self._domain = value
    
    @property
    def task(self):
        return self._task
    
    @task.setter
    def task(self, value):
        self._task = value
    
    @property
    def box(self):
        return self._box
    
    @box.setter
    def box(self, value):
        self._box = value
    
    def __repr__(self):
        info = "Datum(impath={}, label={}, domain={}, task={}, box={})".format(
            self.impath, self.label, self.domain, self.task, self.box
        )
        return info
    

class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []
        # import pdb; pdb.set_trace()
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output
    
    @staticmethod
    def split_dataset_by_label(data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output
    
    
class DatasetBaseMT(DatasetBase):
    
    def __init__(self, taskid2tname, taskid2cname, taskid2desc, taskid2type, taskid2metric, 
                 train_x=None, train_u=None, val=None, test=None):
        
        self.taskid2tname = taskid2tname
        self.taskid2cname = taskid2cname
        self.taskid2desc = taskid2desc
        self.taskid2type = taskid2type
        self.taskid2metric = taskid2metric
        super().__init__(train_x, train_u, val, test)
    
    def get_num_classes(self, data_source=None):
        return [len(x) for x in self.taskid2cname]
    
    def get_lab2cname(self, data_source=None):
        
        lab2cname = {i:x for i,x in enumerate(self.taskid2cname)}
        classnames = self.taskid2cname
        
        return lab2cname, classnames
    
    @property
    def lab2cname(self):
        return self._lab2cname
    
    @property
    def classnames(self):
        return self._classnames
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @staticmethod
    def split_dataset_by_label(data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = OrderedDict()
        
        for item in data_source:
            for l, t in zip(item.label, item.task):
                if (t, l) in output:
                    output[(t, l)].append(item)
                else:
                    output[(t, l)] = [item]
        
        return output