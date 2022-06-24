from collections import namedtuple
from pathlib import Path

# Dataset and results root directory
_SOURCE_FILE_ROOT = Path(__file__).parent.parent.parent / 'Data/source_files'
_BUG_REPORTS_ROOT = Path(__file__).parent.parent.parent / 'Data/bug_reports'
_PICKLE_FILES_ROOT = Path(__file__).parent.parent.parent / 'Data/pickle_files'
RESULTS_ROOT = Path(__file__).parent.parent.parent / 'Results'
RESULTS_ROOT.mkdir(exist_ok=True)

# Init Dataset
Dataset = namedtuple('Dataset', ['name', 'root', 'src', 'bug_repo', 'results'])

# Source codes and bug repositories
aspectj = Dataset(
    'aspectj',
    _PICKLE_FILES_ROOT / 'Aspectj',
    _SOURCE_FILE_ROOT / 'org.aspectj-bug433351',
    _BUG_REPORTS_ROOT / 'AspectJ.txt',
    RESULTS_ROOT / 'Aspectj'
)

birt = Dataset(
    'birt',
    _PICKLE_FILES_ROOT / 'Birt',
    _SOURCE_FILE_ROOT / 'birt-20140211-1400',
    _BUG_REPORTS_ROOT / 'Birt.txt',
    RESULTS_ROOT / 'Birt'
)

swt = Dataset(
    'swt',
    _PICKLE_FILES_ROOT / 'SWT',
    _SOURCE_FILE_ROOT / 'eclipse.platform.swt-xulrunner-31',
    _BUG_REPORTS_ROOT / 'SWT.txt',
    RESULTS_ROOT / 'SWT'
)

ui = Dataset(
    'ui',
    _PICKLE_FILES_ROOT / 'UI',
    _SOURCE_FILE_ROOT / 'eclipse.platform.ui-johna-402445',
    _BUG_REPORTS_ROOT / 'Eclipse_Platform_UI.txt',
    RESULTS_ROOT / 'UI'
)

tomcat = Dataset(
    'tomcat',
    _PICKLE_FILES_ROOT / 'Tomcat',
    _SOURCE_FILE_ROOT / 'tomcat-7.0.51',
    _BUG_REPORTS_ROOT / 'Tomcat.txt',
    RESULTS_ROOT / 'Tomcat'
)

# Select data to run
DATASET = aspectj

# Test
if __name__ == "__main__":
    print(DATASET)