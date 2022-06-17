from collections import namedtuple
from pathlib import Path

# Dataset and results root directory
_SOURCE_FILE_ROOT = Path(__file__).parent.parent.parent / 'Data/source_files'
_BUG_REPORTS_ROOT = Path(__file__).parent.parent.parent / 'Data/bug_reports'
RESULTS_ROOT = Path(__file__).parent.parent.parent / 'Results'
RESULTS_ROOT.mkdir(exist_ok=True)

# Init Dataset
Dataset = namedtuple('Dataset', ['name', 'src', 'bug_reports', 'results'])

# Source codes and bug repositories
aspectj = Dataset(
    'aspectj',
    _SOURCE_FILE_ROOT / 'org.aspectj-bug433351',
    _BUG_REPORTS_ROOT / 'AspectJ.xlsx',
    RESULTS_ROOT / 'Aspectj'
)

birt = Dataset(
    'birt',
    _SOURCE_FILE_ROOT / 'birt-20140211-1400',
    _BUG_REPORTS_ROOT / 'Birt.xlsx',
    RESULTS_ROOT / 'Birt'
)

swt = Dataset(
    'swt',
    _SOURCE_FILE_ROOT / 'eclipse.platform.swt-xulrunner-31',
    _BUG_REPORTS_ROOT / 'SWT.xlsx',
    RESULTS_ROOT / 'SWT'
)

ui = Dataset(
    'ui',
    _SOURCE_FILE_ROOT / 'eclipse.platform.ui-johna-402445',
    _BUG_REPORTS_ROOT / 'Eclipse_Platform_UI.xlsx',
    RESULTS_ROOT / 'UI'
)

tomcat = Dataset(
    'tomcat',
    _SOURCE_FILE_ROOT / 'tomcat-7.0.51',
    _BUG_REPORTS_ROOT / 'Tomcat.xlsx',
    RESULTS_ROOT / 'Tomcat'
)

# Test
if __name__ == "__main__":
    print(aspectj)