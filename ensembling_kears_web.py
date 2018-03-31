#For keras model
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,concatenate
import keras.applications as app

#For Web service
from keras.preprocessing import image
from flask import render_template
from flask import Flask, request
from werkzeug.utils import secure_filename

#Tools
import utils
import os
import numpy as np

#For error image check
import keras.applications.inception_resnet_v2 as inception_resnet_v2
import imagenet_cats_and_dogs as cd_class

model_name = "ensembling_kears"

ensembling_model_list = {
    'DenseNet201': {'model': app.densenet.DenseNet201, 'preprocess': app.densenet.preprocess_input, 'input_size': (224, 224, 3)},  # params 1920   v3 v2
    'ResNet50': {'model': app.resnet50.ResNet50, 'preprocess': app.resnet50.preprocess_input, 'input_size': (224, 224, 3)},  # 2048
    'InceptionResNetV2': {'model': app.inception_resnet_v2.InceptionResNetV2, 'preprocess': app.inception_resnet_v2.preprocess_input, 'input_size': (224, 224, 3)}, # params 1536
    'Xception': {'model': app.xception.Xception, 'preprocess': app.xception.preprocess_input, 'input_size': (299, 299, 3)},  # params 2048
    'InceptionV3': {'model': app.inception_v3.InceptionV3, 'preprocess': app.inception_v3.preprocess_input, 'input_size': (299, 299, 3)},  # params 2048
}

def load_pre_train(model, fuc_preprocess, model_name, add_output_global_average_pooling_flag=True):

    if add_output_global_average_pooling_flag:
        model = Model(model.input, GlobalAveragePooling2D()(model.output))

    return model

base_model_output = []
base_model_input = []

# create image feature by train
for key in ['DenseNet201','ResNet50','InceptionResNetV2','Xception','InceptionV3']:
    property = ensembling_model_list[key]

    # begin get image feature
    model = load_pre_train(model=property['model'](weights='imagenet', include_top=False, input_shape=property['input_size']),
                                      # pooling = "avg",
                                      fuc_preprocess=property['preprocess'],
                                      model_name=key)
    base_model_output.append(model.output)
    base_model_input.append(model.input)
    print("model {} is loaded".format(key))

#merge output
merged = concatenate(base_model_output, axis=1)

#load best model and merge model
best_model = utils.get_best_model(model_name)
final_model = best_model(merged)
final_model = Model(base_model_input, final_model)

#error check model
model_for_error = inception_resnet_v2.InceptionResNetV2(weights='imagenet')

#Just for check error input
def is_cat_or_dog(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_resnet_v2.preprocess_input(x)
    preds = np.array(inception_resnet_v2.decode_predictions(model_for_error.predict(x), top=40)[0])

    is_cat_or_dog = False
    for pred in preds[:, 0]:
        if pred in cd_class.cats or pred in cd_class.dogs:
            is_cat_or_dog = True
            break;
    return is_cat_or_dog

UPLOAD_FOLDER = '/tmp/upload'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')

@app.route('/ajax_process', methods=['POST'])
def ajax_process():
    file = request.files['upload_image']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if is_cat_or_dog(img_path) == False:
        os.remove(img_path)
        return "文件（%s），你确认上传的图片不是外星猫、狗？" % (filename)

    input_x = []
    for key in ['DenseNet201', 'ResNet50', 'InceptionResNetV2', 'Xception', 'InceptionV3']:
        property = ensembling_model_list[key]

        img = image.load_img(img_path, target_size=(property['input_size'][0], property['input_size'][1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = property['preprocess'](x)
        input_x.append(x)

    score = final_model.predict(input_x)

    if score[0][0] >= 0.5:
        return "文件（%s）是狗，得分（满分1）：%.5f"%(filename, score[0][0])
    else:
        return "文件（%s）是猫，得分（满分1）：%.5f"%(filename, 1 - score[0][0])

if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0')
