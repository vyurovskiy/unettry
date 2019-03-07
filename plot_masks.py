import json
import sys
import os
from tqdm import tqdm
from putils._mir_hook import mir

with open('./config.json') as f:
    config = json.load(f)
if not os.path.exists(config['masksP']):
    os.mkdir(config['masksP'])


def draw_masks(dataP, xmlsP, masksP):

    # draw masks with tif and xml
    for name in tqdm(os.listdir(dataP)):
        reader = mir.MultiResolutionImageReader()
        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)
        mr_image = reader.open(os.path.join(dataP + name))
        name = name.split('.tif')[0]
        xml_repository.setSource(xmlsP + name + '.xml')
        xml_repository.load()
        annotation_mask = mir.AnnotationToMask()
        output_path = masksP + name + '_M.tif'
        annotation_mask.convert(
            annotation_list, output_path, mr_image.getDimensions(),
            mr_image.getSpacing()
        )


if __name__ == '__main__':
    draw_masks(config['imagesP'], config['xmlsP'], config['masksP'])
