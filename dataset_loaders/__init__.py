from subprocess import check_output

from images.camvid import CamvidDataset  # noqa
from images.cifar10 import Cifar10Dataset  # noqa
from images.cityscapes import CityscapesDataset  # noqa
from images.isbi_em_stacks import IsbiEmStacksDataset  # noqa
from images.kitti import KITTIdataset  # noqa
from images.mscoco import MSCocoDataset  # noqa
from images.pascalvoc import PascalVOCdataset  # noqa
from images.polyps912 import Polyps912Dataset  # noqa
from images.scene_parsing_MIT import SceneParsingMITDataset  # noqa

from videos.change_detection import ChangeDetectionDataset  # noqa
from videos.davis import DavisDataset  # noqa
from videos.davis2017 import Davis2017Dataset # noqa
from videos.gatech import GatechDataset  # noqa
from videos.movingMNIST import MovingMNISTDataset # noqa
__version__ = check_output('git rev-parse HEAD',
                           shell=True).strip().decode('ascii')
