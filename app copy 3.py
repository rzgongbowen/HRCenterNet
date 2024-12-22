from flask import Flask, render_template, request, jsonify
import torch
from torch.autograd import Variable
import torchvision
from PIL import Image
import numpy as np
from skimage.draw import rectangle_perimeter
import os
from models.HRCenterNet_ca_T2_Bottle2neck import HRCenterNet
import torchvision.transforms.functional as TF
import shutil
from flask import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, session



app = Flask(__name__)
# app = Flask(__name__, static_folder='static', static_url_path='/static')
os.makedirs(r'E:\gbw\HRC\HRCenterNet\static\uploads', exist_ok=True)
app.config['UPLOAD_FOLDER'] = r'E:\gbw\HRC\HRCenterNet\static\uploads'
app.secret_key = '00cce6c5aa8708ff7f2b46332b7beddfc4b2eeac7a3d89c1f65a18490699b4b7'

valid_credentials = {
    'root': '123',
    'user1': '123'
}

input_size = 512
output_size = 256

test_tx = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

@app.route('/result/<filename>')
def get_result_image(filename):
    return send_from_directory('static', filename)

def _nms(img, predict, out_path, filename, nms_score, iou_threshold):
    bbox = []
    score_list = []
    im_draw = np.asarray(TF.resize(img, (img.size[1], img.size[0]))).copy()

    heatmap = predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]

    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:
        row = j // output_size
        col = j - row * output_size

        bias_x = offset_x[row, col] * (img.size[1] / output_size)
        bias_y = offset_y[row, col] * (img.size[0] / output_size)

        width = width_map[row, col] * output_size * (img.size[1] / output_size)
        height = height_map[row, col] * output_size * (img.size[0] / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img.size[1] / output_size) + bias_y
        col = col * (img.size[0] / output_size) + bias_x

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        bbox.append([top, left, bottom, right])

    if len(bbox) > 0:
        # 执行非最大抑制
        _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)),
                                         iou_threshold=iou_threshold)

        print(f'共有{_nms_index.shape[0]}汉字')

        for k in _nms_index:
            top, left, bottom, right = bbox[k]

            start = (int(top), int(left))
            end = (int(bottom), int(right))

            rr, cc = rectangle_perimeter(start, end=end, shape=(img.size[1], img.size[0]))

            im_draw[rr, cc] = (255, 0, 0)

            cropped = img.crop((left, top, right, bottom))

            cropped_out_path = str(out_path) + '\\cropped_out\\' +'\\'+ filename[:-4] +'\\'

            
            os.makedirs(cropped_out_path, exist_ok=True)

            cropped.save(cropped_out_path + filename[:-4] + '_' +'%d.jpg' % k)

    return im_draw, cropped_out_path

def process_image(filepath):
    try:
        model = HRCenterNet()
        checkpoint_path = r'E:\gbw\HRC\HRCenterNet\weight\best.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # 打开图像并转换为RGB模式
        img = Image.open(filepath).convert("RGB")
        image_tensor = test_tx(img)
        image_tensor = image_tensor.unsqueeze_(0)

        with torch.no_grad():
            image_tensor = image_tensor.to(device, dtype=torch.float)
            predict = model(image_tensor)

        print(os.path.basename(filepath))
        
        out_img, cropped_out_path = _nms(img, predict, r'E:\gbw\HRC\HRCenterNet\static', os.path.basename(filepath), nms_score=0.3, iou_threshold=0.1)

        # 列表用于保存所有图片文件的绝对路径
        cropped_files_list = []

        # 遍历目标文件夹
        for root, dirs, files in os.walk(cropped_out_path):
            for file in files:
                # 获取文件的绝对路径
                file_path = os.path.join(root, file)
                
                # 检查文件类型是否为图片文件
                _, extension = os.path.splitext(file_path)
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
                
                # 如果是图片文件，则将其绝对路径添加到列表中
                if extension.lower() in image_extensions:
                    cropped_files_list.append(file_path)


        # 保存处理后的图像到临时目录并返回路径
        temp_output_path = r'E:\gbw\HRC\HRCenterNet\static\temp_output.jpg'
        Image.fromarray(out_img).save(temp_output_path)

        return temp_output_path, cropped_out_path, cropped_files_list  
    except Exception as e:
        # 打印出错信息
        print(f"图像处理失败: {e}")
        raise RuntimeError("图像处理失败，请重试。")

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username in valid_credentials and valid_credentials[username] == password:
        session['logged_in'] = True
        return redirect(url_for('index'))
    else:
        error_message = "Invalid username or password. Please try again."
        return render_template('login.html', error=error_message)

    return render_template('login.html', error=None)

@app.route('/index')
def index():
    if 'logged_in' in session and session['logged_in']:
        return render_template('index.html')
    else:
        return redirect(url_for('login_page'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login_page'))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 处理图像并返回处理后的图像路径
            result_filepath, cropped_out_path, cropped_files_list = process_image(filepath)

            print(f'待预测图片路径：{filepath}')
            print(f'结果图片路径：{result_filepath}')
            print(f'结果图片文件夹路径（分割）：{cropped_out_path}')
            print(cropped_files_list)

            response = {
                'success': True,
                'result': result_filepath,
                'cropped_files_list': cropped_files_list
            }

            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
