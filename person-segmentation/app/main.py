from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from torchvision.models import resnet50, ResNet50_Weights
# from PIL import Image


# Define the FastAPI app
app = FastAPI()

# Initialize weights, transforms and model
weights = ResNet50_Weights.DEFAULT
preprocess_image = weights.transforms()
model = resnet50(weights=weights)


@app.get('/')
async def root():
    return "WELCOME. Go to /docs or /predict or send post request to /predict"


@app.get('/predict', response_class=HTMLResponse)
async def get_image():
    with open('html-sources/predict.html', 'r', encoding='utf-8') as file:
        html = file.read()
    return html


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read image
    img = None  # FIXME

    # Preprocess the input image
    img_transformed = preprocess_image(img)

    # Set model to eval mode
    model.eval()

    return FileResponse('image.png', media_type='image/png')


# def preprocess_image(img_mx_array, model, thresh=0.5):
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


    # Save UploadFile object to image file on disk
    # with open('image.jpg', 'wb') as buffer:
    #     shutil.copyfileobj(file.file, buffer)

    # Convert numpy array to stream
    # pil_image = Image.fromarray(out_img_np_array)
    # stream = io.BytesIO()
    # pil_image.save(stream, format="JPEG")
