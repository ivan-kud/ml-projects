# import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
# from PIL import Image


# Define the FastAPI app
app = FastAPI()

# # Load pretrained model
# model = gluoncv.model_zoo.get_model('mask_rcnn_fpn_resnet101_v1d_coco',
#                                     pretrained=True, root='./models')


# def process_image(img_mx_array, model, thresh=0.5):
#     # Preprocess an image
#     img_mx_array, img_np_array = gluoncv.data.transforms.presets.rcnn.transform_test(img_mx_array)
#
#     # Inference
#     ids, scores, bboxes, masks = [x[0].asnumpy() for x in model(img_mx_array)]
#
#     # Filter by person id
#     person_id = model.classes.index('person')
#     scores = scores[ids[:, 0] == person_id, :]
#     bboxes = bboxes[ids[:, 0] == person_id, :]
#     masks = masks[ids[:, 0] == person_id, :, :]
#     ids = ids[ids[:, 0] == person_id, :]
#
#     # Paint segmentation mask on images directly
#     width, height = img_np_array.shape[1], img_np_array.shape[0]
#     masks, _ = gluoncv.utils.viz.expand_mask(masks, bboxes, (width, height), scores, thresh)
#     out_img_np_array = gluoncv.utils.viz.plot_mask(img_np_array, masks)
#
#     # Make a Figure and add it to canvas
#     dpi = 80
#     figsize = (out_img_np_array.shape[1] / float(dpi), out_img_np_array.shape[0] / float(dpi))
#     fig = Figure(figsize=figsize, dpi=dpi)
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.axis('off')
#
#     # Paint bboxes, scores and classes on images directly
#     ax = gluoncv.utils.viz.plot_bbox(out_img_np_array, bboxes, scores, ids, thresh,
#                                      model.classes, ax=ax, linewidth=2.0, fontsize=8)
#
#     fig.savefig('image.png')


@app.get('/')
async def root():
    return "WELCOME. Go to /docs or /predict or send post request to /predict"


@app.get('/predict', response_class=HTMLResponse)
async def get_image():
    with open('html-sources/predict.html', 'r') as file:
        html = file.read()
    return html


# @app.post('/predict')
# async def predict(file: UploadFile = File(...)):
#     # Convert file to mxnet array
#     file_contents = await file.read()
#     stream = io.BytesIO(file_contents)
#     pil_image = Image.open(stream)
#     mxnet_array = mxnet.nd.array(pil_image)
#
#     process_image(mxnet_array, model, 0.5)
#
#     return FileResponse('image.png', media_type='image/png')




    # Save UploadFile object to image file on disk
    # with open('image.jpg', 'wb') as buffer:
    #     shutil.copyfileobj(file.file, buffer)

    # Convert numpy array to stream
    # pil_image = Image.fromarray(out_img_np_array)
    # stream = io.BytesIO()
    # pil_image.save(stream, format="JPEG")
