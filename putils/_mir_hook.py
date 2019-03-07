import sys
# https://github.com/computationalpathologygroup/ASAP/releases
# Automated Slide Analysis Platform (ASAP) is an open source platform
if 'D:\\ACDC_LUNG_HISTOPATHOLOGY\\ASAP 1.9\\bin' not in sys.path:
    sys.path.append('D:\\ACDC_LUNG_HISTOPATHOLOGY\\ASAP 1.9\\bin')
import multiresolutionimageinterface as mir  # noqa pylint: disable=import-error
