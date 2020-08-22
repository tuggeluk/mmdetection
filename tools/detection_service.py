import numpy as np
from flask import Flask, request, send_from_directory
from PIL import Image
import json
import torch
import mmcv


UPLOAD_FOLDER = './Patches/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
model = None

class_names = ('brace', 'ledgerLine', 'repeatDot', 'segno', 'coda', 'clefG', 'clefCAlto', 'clefCTenor', 'clefF', 'clefUnpitchedPercussion', 'clef8', 'clef15', 'timeSig0', 'timeSig1', 'timeSig2', 'timeSig3', 'timeSig4', 'timeSig5', 'timeSig6', 'timeSig7', 'timeSig8', 'timeSig9', 'timeSigCommon', 'timeSigCutCommon', 'noteheadBlackOnLine', 'noteheadBlackOnLineSmall', 'noteheadBlackInSpace', 'noteheadBlackInSpaceSmall', 'noteheadHalfOnLine', 'noteheadHalfOnLineSmall', 'noteheadHalfInSpace', 'noteheadHalfInSpaceSmall', 'noteheadWholeOnLine', 'noteheadWholeOnLineSmall', 'noteheadWholeInSpace', 'noteheadWholeInSpaceSmall', 'noteheadDoubleWholeOnLine', 'noteheadDoubleWholeOnLineSmall', 'noteheadDoubleWholeInSpace', 'noteheadDoubleWholeInSpaceSmall', 'augmentationDot', 'stem', 'tremolo1', 'tremolo2', 'tremolo3', 'tremolo4', 'tremolo5', 'flag8thUp', 'flag8thUpSmall', 'flag16thUp', 'flag32ndUp', 'flag64thUp', 'flag128thUp', 'flag8thDown', 'flag8thDownSmall', 'flag16thDown', 'flag32ndDown', 'flag64thDown', 'flag128thDown', 'accidentalFlat', 'accidentalFlatSmall', 'accidentalNatural', 'accidentalNaturalSmall', 'accidentalSharp', 'accidentalSharpSmall', 'accidentalDoubleSharp', 'accidentalDoubleFlat', 'keyFlat', 'keyNatural', 'keySharp', 'articAccentAbove', 'articAccentBelow', 'articStaccatoAbove', 'articStaccatoBelow', 'articTenutoAbove', 'articTenutoBelow', 'articStaccatissimoAbove', 'articStaccatissimoBelow', 'articMarcatoAbove', 'articMarcatoBelow', 'fermataAbove', 'fermataBelow', 'caesura', 'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th', 'rest32nd', 'rest64th', 'rest128th', 'restHNr', 'dynamicP', 'dynamicM', 'dynamicF', 'dynamicS', 'dynamicZ', 'dynamicR', 'graceNoteAcciaccaturaStemUp', 'graceNoteAppoggiaturaStemUp', 'graceNoteAcciaccaturaStemDown', 'graceNoteAppoggiaturaStemDown', 'ornamentTrill', 'ornamentTurn', 'ornamentTurnInverted', 'ornamentMordent', 'stringsDownBow', 'stringsUpBow', 'arpeggiato', 'keyboardPedalPed', 'keyboardPedalUp', 'tuplet3', 'tuplet6', 'fingering0', 'fingering1', 'fingering2', 'fingering3', 'fingering4', 'fingering5', 'slur', 'beam', 'tie', 'restHBar', 'dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin', 'tuplet1', 'tuplet2', 'tuplet4', 'tuplet5', 'tuplet7', 'tuplet8', 'tuplet9', 'tupletBracket', 'staff', 'ottavaBracket')

app = Flask(__name__)


@app.route('/')
def hello_world():
    message = 'Welcome to the classifier'
    return message

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        print(request.headers)
        file = request.files['image']
        if file and allowed_file(file.filename):

            pic = Image.open(file).convert('RGB')
            img = np.asarray(pic)

            result = inference_detector(model, img)
            detect_list = []
            for cla, bboxes in enumerate(result):
                for bbox in bboxes:
                    bbox = [int(x) for x in bbox]
                    bbox.append(bbox[4])
                    bbox[4] = class_names[cla]
                    detect_list.append(bbox)

            detect_dict = dict(bounding_boxes = detect_list)
            print(json.dumps(detect_dict))
            return json.dumps(detect_dict)
        else:
            return 'Unsupported filetype'
    return
    '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=image_patch>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_img(pixels):
    data = {}
    data['img_metas'] = None
    data['img'] = pixels


    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result




if __name__ == '__main__':
    from mmdet.apis import init_detector, inference_detector

    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

    config_path = "configs/DeepScoresBaselines/faster_rcnn_v2/faster_rcnn_hrnetv2p_w32_1x_coco.py"
    pretrained_path = "work_dirs/faster_rcnn_hrnetv2p_w32_1x_coco/epoch_140.pth"
    # build the model from a config file and a checkpoint file
    model = init_detector(config_path, pretrained_path, device='cuda:0')

    # test a single image and show the results
    #img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    #img = mmcv.imread("/home/tugg/Documents/Detection_Service/demo/Dvorak_Slawischer_Tanz_8_Musicalion_n.png")


    app.run(host='0.0.0.0')